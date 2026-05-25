// tinyC parser — recursive descent. Produces an AST defined in tinyc.h.

#include "tinyc.h"

#include <base/arena.h>
#include <base/format.h>
#include <base/io.h>
#include <base/string.h>

// One entry in the typedef map. Module-scope only.
typedef struct Typedef {
    string name;
    Type   ty;
    struct Typedef *next;
} Typedef;

// One entry in the enumerator map. Module-scope only. Enumerators are
// always i32 constants and live in the same value namespace as variables
// and functions (lookup falls through to them after locals/globals miss).
typedef struct Enumerator {
    string  name;
    int64_t value;
    struct Enumerator *next;
} Enumerator;

typedef struct {
    Arena *arena;
    TcTok *toks;
    size_t n;
    size_t i;
    Typedef *typedefs;
    Enumerator *enums;
    int err_count;
    struct Program *prog;
    // When true, `long` / `size_t` / `intptr_t` / `uintptr_t` /
    // `ptrdiff_t` / `ssize_t` are 32-bit (wasm32-wasi ABI). When
    // false (the default for native host targets) they're 64-bit.
    // `long long` and `int64_t` are always 64-bit regardless.
    bool target_wasm32;
    // Name of the function currently being parsed (set by parse_func
    // around the body). Used to mangle function-local `static`
    // variables into unique module-scope global names.
    string cur_func_name;
    int    static_local_counter;
} P;

static TcTok cur(P *p) { return p->toks[p->i]; }
static TcTok peek(P *p, size_t k) { return p->toks[p->i + k < p->n ? p->i + k : p->n - 1]; }

static Typedef *typedef_lookup(P *p, string name) {
    for (Typedef *t = p->typedefs; t; t = t->next) {
        if (str_eq(t->name, name)) return t;
    }
    return NULL;
}

static Enumerator *enum_lookup(P *p, string name) {
    for (Enumerator *e = p->enums; e; e = e->next) {
        if (str_eq(e->name, name)) return e;
    }
    return NULL;
}

// Best-effort constant-folder used by enum-body parsing. Returns true iff
// `ex` is a constant integer expression whose value can be computed using
// only int literals and previously-declared enumerators. Recognizes the
// common arithmetic, bitwise, shift, comparison and ternary forms; runs
// before the emitter's `ast_fold_int`, so it intentionally only handles
// the subset that doesn't need scope/type information.
static bool parse_const_eval(P *p, Expr *ex, int64_t *out) {
    if (!ex) return false;
    switch (ex->kind) {
        case EX_INT:
            *out = ex->int_value;
            return true;
        case EX_VAR: {
            Enumerator *en = enum_lookup(p, ex->name);
            if (!en) return false;
            *out = en->value;
            return true;
        }
        case EX_UN: {
            int64_t a;
            if (!parse_const_eval(p, ex->lhs, &a)) return false;
            switch (ex->op) {
                case OP_NEG:  *out = -a;        return true;
                case OP_BNOT: *out = ~a;        return true;
                case OP_NOT:  *out = (a == 0);  return true;
                default: return false;
            }
        }
        case EX_BIN: {
            int64_t a, b;
            if (!parse_const_eval(p, ex->lhs, &a)) return false;
            if (!parse_const_eval(p, ex->rhs, &b)) return false;
            switch (ex->op) {
                case OP_ADD:  *out = a + b; return true;
                case OP_SUB:  *out = a - b; return true;
                case OP_MUL:  *out = a * b; return true;
                case OP_DIV:  if (b == 0) return false; *out = a / b; return true;
                case OP_MOD:  if (b == 0) return false; *out = a % b; return true;
                case OP_BAND: *out = a & b; return true;
                case OP_BOR:  *out = a | b; return true;
                case OP_BXOR: *out = a ^ b; return true;
                case OP_SHL:  *out = a << b; return true;
                case OP_SHR:  *out = a >> b; return true;
                case OP_LT:   *out = a <  b; return true;
                case OP_LE:   *out = a <= b; return true;
                case OP_GT:   *out = a >  b; return true;
                case OP_GE:   *out = a >= b; return true;
                case OP_EQ:   *out = a == b; return true;
                case OP_NE:   *out = a != b; return true;
                case OP_AND:  *out = (a != 0 && b != 0); return true;
                case OP_OR:   *out = (a != 0 || b != 0); return true;
                default: return false;
            }
        }
        case EX_TERNARY: {
            int64_t c;
            if (!parse_const_eval(p, ex->lhs, &c)) return false;
            Expr *pick = c ? ex->rhs : ex->lvalue;
            return parse_const_eval(p, pick, out);
        }
        default:
            return false;
    }
}

static void perror_at(P *p, int line, string msg) {
    println(str_lit("tinyc parse error at line {}: {}"), (int64_t)line, msg);
    p->err_count++;
}

static bool accept(P *p, TcTokKind k) {
    if (cur(p).kind == k) { p->i++; return true; }
    return false;
}
static bool expect(P *p, TcTokKind k, string what) {
    if (accept(p, k)) return true;
    perror_at(p, cur(p).line, what);
    return false;
}

static Expr *new_expr(P *p, ExprKind k, int line) {
    Expr *e = arena_new(p->arena, Expr);
    *e = (Expr){0};
    e->kind = k;
    e->line = line;
    VecExprPtr_reserve(p->arena, &e->args, 4);
    return e;
}
static Stmt *new_stmt(P *p, StmtKind k, int line) {
    Stmt *s = arena_new(p->arena, Stmt);
    *s = (Stmt){0};
    s->kind = k;
    s->line = line;
    VecStmtPtr_reserve(p->arena, &s->then_body, 4);
    VecStmtPtr_reserve(p->arena, &s->else_body, 4);
    VecStmtPtr_reserve(p->arena, &s->while_body, 4);
    VecStmtPtr_reserve(p->arena, &s->for_body, 4);
    VecExprPtr_reserve(p->arena, &s->for_steps, 2);
    VecStmtPtr_reserve(p->arena, &s->block_body, 4);
    return s;
}

static Expr *parse_expr(P *p);
static bool parse_sig_type(P *p, Type *out);

// Holder for the subset of `__attribute__((...))` annotations that
// affect codegen. Empty strings mean "not present". Used by the
// declaration parser to collect attributes that appear before, after,
// or in the middle of a declaration and then attach them to the
// resulting Func.
typedef struct {
    string import_module;
    string import_name;
    string export_name;
} AttrInfo;

// Parse zero or more `__attribute__((...))` attribute lists. Recognized
// attribute names: `import_module("...")`, `import_name("...")`,
// `export_name("...")` (and the gcc-style `__import_module__` /
// `__import_name__` / `__export_name__` underscore variants used by
// corec/platform/platform_wasm.c). Unknown attributes are silently
// skipped via balanced-paren scanning so that arbitrary GCC/Clang
// annotations like `__attribute__((weak))` or `__attribute__((noinline))`
// are accepted without modification of the source.
//
// `out` may be NULL when the caller doesn't care about the semantic
// effect (e.g. attributes on a struct definition that we don't model).
// In that case the attributes are still parsed and skipped so we can
// continue past them.
static bool attr_name_eq(string s, const char *bare) {
    string b = str_from_cstr_view((char *)bare);
    if (str_eq(s, b)) return true;
    // Accept the `__name__` underscore-wrapped GCC spelling as well.
    if (s.size != b.size + 4) return false;
    if (s.str[0] != '_' || s.str[1] != '_') return false;
    if (s.str[s.size - 2] != '_' || s.str[s.size - 1] != '_') return false;
    for (size_t i = 0; i < b.size; i++) {
        if (s.str[i + 2] != b.str[i]) return false;
    }
    return true;
}
static void parse_attributes(P *p, AttrInfo *out) {
    while (cur(p).kind == TC_TK_KW_ATTRIBUTE) {
        p->i++;
        // `__attribute__` must be followed by `((` and end with `))`.
        if (!expect(p, TC_TK_LPAREN, str_lit("expected '(' after __attribute__"))) return;
        if (!expect(p, TC_TK_LPAREN, str_lit("expected '((' after __attribute__"))) return;
        // Each attribute is either an identifier or
        // identifier(args...). Multiple are comma-separated.
        // Identifier `()` is also valid (no args).
        for (;;) {
            // Empty attribute list `()` — bail out.
            if (cur(p).kind == TC_TK_RPAREN) break;
            // An attribute name is an identifier; accept keywords like
            // `const` and `static` as identifiers in this position too
            // (gcc allows things like `__attribute__((const))`).
            TcTok nm = cur(p);
            string aname = nm.text;
            if (nm.kind != TC_TK_IDENT && aname.size == 0) {
                // Skip whatever token this is.
                p->i++;
            } else {
                p->i++;
            }
            // Optional `(args)`.
            if (accept(p, TC_TK_LPAREN)) {
                // Capture the first string-literal argument for the
                // attributes we care about. tinyc's lexer includes a
                // trailing NUL in the literal's text (so string-literal
                // values stored in `.rodata` are NUL-terminated); strip
                // it here so the attribute value is a clean view of the
                // string contents.
                string str_arg = (string){0};
                if (cur(p).kind == TC_TK_STRING_LIT) {
                    str_arg = cur(p).text;
                    if (str_arg.size > 0 &&
                        str_arg.str[str_arg.size - 1] == '\0') {
                        str_arg.size--;
                    }
                }
                // Skip the entire argument list, tracking parens.
                int depth = 1;
                while (depth > 0 && cur(p).kind != TC_TK_EOF) {
                    if (cur(p).kind == TC_TK_LPAREN) depth++;
                    else if (cur(p).kind == TC_TK_RPAREN) { depth--; if (depth == 0) break; }
                    p->i++;
                }
                expect(p, TC_TK_RPAREN, str_lit("expected ')' in __attribute__"));
                if (out) {
                    if (attr_name_eq(aname, "import_module")) out->import_module = str_arg;
                    else if (attr_name_eq(aname, "import_name")) out->import_name = str_arg;
                    else if (attr_name_eq(aname, "export_name")) out->export_name = str_arg;
                }
            }
            if (!accept(p, TC_TK_COMMA)) break;
        }
        expect(p, TC_TK_RPAREN, str_lit("expected ')' in __attribute__"));
        expect(p, TC_TK_RPAREN, str_lit("expected '))' in __attribute__"));
    }
}

// Parse the contents of `[ ... ]` as an array-length specifier. The
// caller has already consumed the opening `[`. Consumes the closing `]`.
// Behavior:
//   - Empty `[]`: writes 0 to *out_len and NULL to *out_expr; the caller
//     must subsequently fill in the length from an aggregate initializer.
//   - `[ INT_LIT ]`: writes the literal value to *out_len.
//   - `[ <const-expr> ]`: writes 0 to *out_len and the parsed expression
//     to *out_expr; the emitter folds it at the alloca site (via
//     ast_fold_int with scope, so `sizeof(x) / sizeof(x[0])` works).
static void parse_array_len_bracket(P *p, int64_t *out_len, Expr **out_expr) {
    *out_len = 0;
    *out_expr = NULL;
    if (cur(p).kind == TC_TK_RBRACK) {
        p->i++;  // ']'
        return;
    }
    if (cur(p).kind == TC_TK_INT_LIT && peek(p, 1).kind == TC_TK_RBRACK) {
        TcTok lit = cur(p);
        p->i++;
        p->i++;  // ']'
        *out_len = lit.int_value;
        return;
    }
    Expr *ex = parse_expr(p);
    expect(p, TC_TK_RBRACK, str_lit("expected ']'"));
    *out_expr = ex;
}
static bool parse_abstract_type(P *p, Type *out);
static bool try_parse_fnptr_suffix(P *p, Type *ty, string *out_name);
static void parse_enum_decl_top(P *p);
static void parse_enum_body(P *p);
static void parse_typedef_decl(P *p);
static bool struct_def_equal(StructDef *a, StructDef *b);
static StructDef *parse_struct_def(P *p);

// Aggregate initializer: `{ v0, v1, ... }` or `{ .f1 = v0, .f2 = v1, ... }`.
// Builds an EX_COMPOUND whose cast_type is the decl's type (struct or
// array). Caller has already consumed the `=` token. Returns NULL when
// the next token is not `{` (caller should fall back to parse_expr).
static Expr *parse_aggregate_init(P *p, Type decl_type) {
    if (cur(p).kind != TC_TK_LBRACE) return NULL;
    int line = cur(p).line;
    p->i++;  // '{'
    Expr *e = new_expr(p, EX_COMPOUND, line);
    e->cast_type = decl_type;
    // Collect (name, value) pairs in a temporary array; allocate the
    // parallel name array once at the end.
    string *names = NULL;
    size_t nn = 0, cap = 0;
    bool any_designated = false;
    if (cur(p).kind != TC_TK_RBRACE) {
        for (;;) {
            string fname = (string){0};
            if (cur(p).kind == TC_TK_DOT) {
                p->i++;
                TcTok ft = cur(p);
                expect(p, TC_TK_IDENT, str_lit("expected field name after '.'"));
                expect(p, TC_TK_ASSIGN, str_lit("expected '=' in designated initializer"));
                fname = ft.text;
                any_designated = true;
            }
            Expr *v;
            if (cur(p).kind == TC_TK_LBRACE) {
                // Nested aggregate initializer (e.g. `{ {a,b}, c }` for a
                // struct field that is itself a struct or array). The
                // element type isn't known here without resolving the
                // surrounding struct definition, so flag the type as
                // TY_VOID (a sentinel: void cannot otherwise appear as a
                // cast_type). The emitter looks up the field type by
                // position / designated name and fills it in before
                // emitting the inner compound.
                Type unset = (Type){0};
                unset.kind = TY_VOID;
                v = parse_aggregate_init(p, unset);
            } else {
                v = parse_expr(p);
            }
            VecExprPtr_push_back(p->arena, &e->args, v);
            if (nn == cap) {
                size_t ncap = cap ? cap * 2 : 4;
                string *nb = arena_new_array(p->arena, string, ncap);
                for (size_t k = 0; k < nn; k++) nb[k] = names[k];
                names = nb; cap = ncap;
            }
            names[nn++] = fname;
            if (cur(p).kind == TC_TK_COMMA) {
                p->i++;
                if (cur(p).kind == TC_TK_RBRACE) break;
                continue;
            }
            break;
        }
    }
    expect(p, TC_TK_RBRACE, str_lit("expected '}' in initializer"));
    if (any_designated) {
        e->compound_field_names = names;
    }
    return e;
}


// `const` is accepted as a parser-level qualifier and silently dropped:
// it does not affect any AST type. Skipped before a base type, after `*`
// in pointer declarators, and (via these helpers) wherever a type can
// appear (return type, param, sizeof/cast, typedef target, struct field,
// local decl).
// Parser-level type qualifiers and storage-class specifiers that have no
// effect on the AST and are silently dropped wherever a type can appear:
//   - `const`   (qualifier)
//   - `static`  (storage class — no linkage change in tinyC)
//   - `inline`  (no inlining performed)
// Any combination, in any order, even repetitions, is accepted.
static void skip_const(P *p) {
    while (cur(p).kind == TC_TK_KW_CONST ||
           cur(p).kind == TC_TK_KW_STATIC ||
           cur(p).kind == TC_TK_KW_INLINE) {
        p->i++;
    }
}

// `static` and `inline` are parser-only storage-class qualifiers. Any
// combination (`static`, `inline`, `static inline`, `inline static`,
// even repetitions) is silently accepted at any position where a type
// can appear. They do not affect the AST or codegen — names retain
// external linkage and inline functions are not substituted.
static void skip_storage_class(P *p) {
    while (cur(p).kind == TC_TK_KW_STATIC || cur(p).kind == TC_TK_KW_INLINE) {
        p->i++;
    }
}

// postfix := (...) | [expr] | .ident
static Expr *parse_postfix(P *p, Expr *base) {
    for (;;) {
        if (cur(p).kind == TC_TK_LBRACK) {
            int line = cur(p).line;
            p->i++;
            Expr *idx = parse_expr(p);
            expect(p, TC_TK_RBRACK, str_lit("expected ']'"));
            Expr *e = new_expr(p, EX_INDEX, line);
            e->lhs = base; e->rhs = idx;
            base = e;
            continue;
        }
        if (cur(p).kind == TC_TK_DOT) {
            int line = cur(p).line;
            p->i++;
            TcTok name = cur(p);
            expect(p, TC_TK_IDENT, str_lit("expected field name"));
            Expr *e = new_expr(p, EX_FIELD, line);
            e->lhs = base; e->name = name.text;
            base = e;
            continue;
        }
        if (cur(p).kind == TC_TK_ARROW) {
            // p->f desugars to (*p).f
            int line = cur(p).line;
            p->i++;
            TcTok name = cur(p);
            expect(p, TC_TK_IDENT, str_lit("expected field name"));
            Expr *deref = new_expr(p, EX_DEREF, line);
            deref->lhs = base;
            Expr *e = new_expr(p, EX_FIELD, line);
            e->lhs = deref; e->name = name.text;
            base = e;
            continue;
        }
        if (cur(p).kind == TC_TK_PLUSPLUS || cur(p).kind == TC_TK_MINUSMINUS) {
            // Postfix ++/--: yield the value of `base` *before* the step
            // (proper C semantics). The desugared assign carries the +/-
            // operation; emit notices `is_post_step` and arranges for the
            // expression to evaluate to the old value.
            BinOp op = (cur(p).kind == TC_TK_PLUSPLUS) ? OP_ADD : OP_SUB;
            int line = cur(p).line;
            p->i++;
            Expr *one = new_expr(p, EX_INT, line);
            one->int_value = 1;
            Expr *bin = new_expr(p, EX_BIN, line);
            bin->op = op; bin->lhs = base; bin->rhs = one;
            Expr *as = new_expr(p, EX_ASSIGN, line);
            as->lvalue = base; as->rhs_assign = bin;
            as->is_post_step = true;
            base = as;
            continue;
        }
        if (cur(p).kind == TC_TK_LPAREN) {
            // Indirect call through a callable expression. Examples:
            //   (*fp)(args)   — fp is a TY_PTR_PTR of a function pointer
            //   funcs[i](x)   — array-of-fnptr (not yet supported)
            // The callee expression is stored in `lhs`; `callee` (a string)
            // stays empty so the EX_CALL emit path knows to evaluate `lhs`
            // as a !llvm.ptr / fnptr value.
            int line = cur(p).line;
            p->i++;
            Expr *e = new_expr(p, EX_CALL, line);
            e->lhs = base;
            if (cur(p).kind != TC_TK_RPAREN) {
                for (;;) {
                    Expr *a = parse_expr(p);
                    VecExprPtr_push_back(p->arena, &e->args, a);
                    if (!accept(p, TC_TK_COMMA)) break;
                }
            }
            expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
            base = e;
            continue;
        }
        break;
    }
    return base;
}

// primary := INT_LIT | FLOAT_LIT | IDENT | IDENT '(' args ')' | '(' expr ')' | unary
static Expr *parse_primary(P *p) {
    TcTok t = cur(p);
    if (t.kind == TC_TK_INT_LIT) {
        p->i++;
        Expr *e = new_expr(p, EX_INT, t.line);
        e->int_value = t.int_value;
        e->is_i64 = t.is_i64;
        e->is_long_long = t.is_long_long;
        // On wasm32, a single-L suffix (`long`) is 32-bit, not 64-bit. Only
        // `LL` (or having no L at all) is target-independent.
        if (e->is_i64 && !e->is_long_long && p->target_wasm32) e->is_i64 = false;
        return e;
    }
    if (t.kind == TC_TK_FLOAT_LIT) {
        p->i++;
        Expr *e = new_expr(p, EX_FLOAT, t.line);
        e->float_value = t.float_value;
        e->is_f64 = t.is_f64;
        return e;
    }
    if (t.kind == TC_TK_KW_NULL) {
        p->i++;
        Expr *e = new_expr(p, EX_NULL, t.line);
        return e;
    }
    if (t.kind == TC_TK_KW_SIZEOF) {
        p->i++;
        Expr *e = new_expr(p, EX_SIZEOF, t.line);
        // C also accepts `sizeof <unary-expression>` (no parens). In that
        // form the operand is always an expression, never a type-name.
        if (cur(p).kind != TC_TK_LPAREN) {
            e->lhs = parse_primary(p);
            e->sizeof_is_expr = true;
            return e;
        }
        p->i++;  // consume '('
        // Disambiguate `sizeof(<type>)` vs `sizeof(<expr>)`. A type-name
        // starts with a base-type keyword, `const`, `enum`, or a typedef
        // identifier. Anything else is parsed as an expression.
        TcTokKind nxt = cur(p).kind;
        bool looks_like_type =
            nxt == TC_TK_KW_INT || nxt == TC_TK_KW_FLOAT || nxt == TC_TK_KW_DOUBLE ||
            nxt == TC_TK_KW_CHAR || nxt == TC_TK_KW_VOID ||
            nxt == TC_TK_KW_STRUCT || nxt == TC_TK_KW_UNION || nxt == TC_TK_KW_ENUM ||
            nxt == TC_TK_KW_CONST || nxt == TC_TK_KW_LONG ||
            nxt == TC_TK_KW_SIGNED || nxt == TC_TK_KW_UNSIGNED ||
            nxt == TC_TK_KW_SHORT || nxt == TC_TK_KW_BOOL ||
            (nxt == TC_TK_IDENT && typedef_lookup(p, cur(p).text) != NULL);
        if (looks_like_type) {
            Type ty = {0};
            if (!parse_abstract_type(p, &ty)) {
                perror_at(p, cur(p).line, str_lit("expected type in sizeof(...)"));
            }
            e->cast_type = ty;
        } else {
            e->lhs = parse_expr(p);
            e->sizeof_is_expr = true;
        }
        expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
        return e;
    }
    if (t.kind == TC_TK_STRING_LIT) {
        p->i++;
        // Adjacent string-literal concatenation: "abc" "def" "ghi" -> "abcdefghi".
        // Note: lexer stores each literal's text with a trailing NUL counted
        // in size, so we strip the NUL from all but the last segment.
        if (cur(p).kind == TC_TK_STRING_LIT) {
            size_t total = (t.text.size > 0 ? t.text.size - 1 : 0);
            size_t k = p->i;
            size_t last_size = 0;
            while (k < p->n && p->toks[k].kind == TC_TK_STRING_LIT) {
                last_size = p->toks[k].text.size;
                total += (last_size > 0 ? last_size - 1 : 0);
                k++;
            }
            total += 1;  // single trailing NUL
            char *buf = arena_new_array(p->arena, char, total);
            size_t off = 0;
            size_t body0 = (t.text.size > 0 ? t.text.size - 1 : 0);
            for (size_t j = 0; j < body0; j++) buf[off++] = t.text.str[j];
            while (cur(p).kind == TC_TK_STRING_LIT) {
                TcTok nx = cur(p);
                size_t body = (nx.text.size > 0 ? nx.text.size - 1 : 0);
                for (size_t j = 0; j < body; j++) buf[off++] = nx.text.str[j];
                p->i++;
            }
            buf[off++] = '\0';
            t.text = (string){.str = buf, .size = off};
        }
        Expr *e = new_expr(p, EX_STR, t.line);
        e->name = t.text;
        return e;
    }
    if (t.kind == TC_TK_LPAREN) {
        // Could be a cast `(T)expr` or a parenthesized expression.
        TcTokKind nxt = peek(p, 1).kind;
        bool is_cast = (nxt == TC_TK_KW_INT || nxt == TC_TK_KW_FLOAT || nxt == TC_TK_KW_DOUBLE ||
                        nxt == TC_TK_KW_STRUCT || nxt == TC_TK_KW_UNION || nxt == TC_TK_KW_CHAR ||
                        nxt == TC_TK_KW_CONST || nxt == TC_TK_KW_LONG ||
                        nxt == TC_TK_KW_SIGNED || nxt == TC_TK_KW_UNSIGNED ||
                        nxt == TC_TK_KW_SHORT || nxt == TC_TK_KW_BOOL ||
                        nxt == TC_TK_KW_VOID);
        if (!is_cast && nxt == TC_TK_IDENT &&
            typedef_lookup(p, peek(p, 1).text)) {
            is_cast = true;
        }
        if (is_cast) {
            p->i++;  // consume '('
            Type ty = {0};
            if (!parse_abstract_type(p, &ty)) {
                perror_at(p, cur(p).line, str_lit("expected type in cast"));
            }
            expect(p, TC_TK_RPAREN, str_lit("expected ')' in cast"));
            if (cur(p).kind == TC_TK_LBRACE) {
                // Compound literal: (T){ v0, v1, ... } or (T){.f = v, ...}
                p->i++;  // consume '{'
                Expr *e = new_expr(p, EX_COMPOUND, t.line);
                e->cast_type = ty;
                string *names = NULL;
                size_t nn = 0, cap = 0;
                bool any_designated = false;
                if (cur(p).kind != TC_TK_RBRACE) {
                    for (;;) {
                        string fname = (string){0};
                        if (cur(p).kind == TC_TK_DOT) {
                            p->i++;
                            TcTok ft = cur(p);
                            expect(p, TC_TK_IDENT, str_lit("expected field name after '.'"));
                            expect(p, TC_TK_ASSIGN, str_lit("expected '=' in designated initializer"));
                            fname = ft.text;
                            any_designated = true;
                        }
                        Expr *v = parse_expr(p);
                        VecExprPtr_push_back(p->arena, &e->args, v);
                        if (nn == cap) {
                            size_t ncap = cap ? cap * 2 : 4;
                            string *nb = arena_new_array(p->arena, string, ncap);
                            for (size_t k = 0; k < nn; k++) nb[k] = names[k];
                            names = nb; cap = ncap;
                        }
                        names[nn++] = fname;
                        if (cur(p).kind == TC_TK_COMMA) {
                            p->i++;
                            if (cur(p).kind == TC_TK_RBRACE) break;
                            continue;
                        }
                        break;
                    }
                }
                expect(p, TC_TK_RBRACE,
                       str_lit("expected '}' in compound literal"));
                if (any_designated) e->compound_field_names = names;
                return e;
            }
            Expr *operand = parse_primary(p);
            Expr *e = new_expr(p, EX_CAST, t.line);
            e->cast_type = ty;
            e->lhs = operand;
            return e;
        }
        p->i++;
        Expr *e = parse_expr(p);
        expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
        return parse_postfix(p, e);
    }
    if (t.kind == TC_TK_MINUS) {
        p->i++;
        Expr *e = new_expr(p, EX_UN, t.line);
        e->op = OP_NEG;
        e->lhs = parse_primary(p);
        return e;
    }
    if (t.kind == TC_TK_BANG) {
        p->i++;
        Expr *e = new_expr(p, EX_UN, t.line);
        e->op = OP_NOT;
        e->lhs = parse_primary(p);
        return e;
    }
    if (t.kind == TC_TK_TILDE) {
        p->i++;
        Expr *e = new_expr(p, EX_UN, t.line);
        e->op = OP_BNOT;
        e->lhs = parse_primary(p);
        return e;
    }
    if (t.kind == TC_TK_PLUSPLUS || t.kind == TC_TK_MINUSMINUS) {
        // Prefix ++/--: rewrite to `lhs = lhs ± 1`.
        BinOp op = (t.kind == TC_TK_PLUSPLUS) ? OP_ADD : OP_SUB;
        p->i++;
        Expr *target = parse_primary(p);
        Expr *one = new_expr(p, EX_INT, t.line);
        one->int_value = 1;
        Expr *bin = new_expr(p, EX_BIN, t.line);
        bin->op = op; bin->lhs = target; bin->rhs = one;
        Expr *as = new_expr(p, EX_ASSIGN, t.line);
        as->lvalue = target; as->rhs_assign = bin;
        return as;
    }
    if (t.kind == TC_TK_AMP) {
        p->i++;
        Expr *e = new_expr(p, EX_ADDR, t.line);
        e->lhs = parse_primary(p);
        return e;
    }
    if (t.kind == TC_TK_STAR) {
        p->i++;
        Expr *e = new_expr(p, EX_DEREF, t.line);
        e->lhs = parse_primary(p);
        return e;
    }
    if (t.kind == TC_TK_KW_GENERIC) {
        p->i++;
        expect(p, TC_TK_LPAREN, str_lit("expected '(' after _Generic"));
        Expr *ge = new_expr(p, EX_GENERIC, t.line);
        ge->lhs = parse_expr(p);
        ge->generic_default_index = -1;
        // Collect entries. Types go into a parallel array; result exprs
        // go into ge->args (already reserved by new_expr).
        int cap = 0;
        Type *types = NULL;
        int n = 0;
        while (accept(p, TC_TK_COMMA)) {
            int line = cur(p).line;
            Type ty = {0};
            bool is_default = false;
            if (cur(p).kind == TC_TK_KW_DEFAULT) {
                p->i++;
                is_default = true;
            } else if (!parse_abstract_type(p, &ty)) {
                perror_at(p, cur(p).line,
                    str_lit("expected type or 'default' in _Generic"));
            }
            expect(p, TC_TK_COLON, str_lit("expected ':' in _Generic entry"));
            Expr *val = parse_expr(p);
            if (n >= cap) {
                int ncap = cap == 0 ? 4 : cap * 2;
                Type *nt = arena_alloc(p->arena, sizeof(Type) * (size_t)ncap);
                for (int j = 0; j < n; j++) nt[j] = types[j];
                types = nt;
                cap = ncap;
            }
            types[n] = ty;
            VecExprPtr_push_back(p->arena, &ge->args, val);
            if (is_default) ge->generic_default_index = n;
            n++;
            (void)line;
        }
        ge->generic_types = types;
        expect(p, TC_TK_RPAREN, str_lit("expected ')' in _Generic"));
        return parse_postfix(p, ge);
    }
    if (t.kind == TC_TK_IDENT) {
        p->i++;
        if (cur(p).kind == TC_TK_LPAREN) {
            // Built-in `va_arg(ap, T)`: second argument is a TYPE-NAME, not
            // an expression. Lower to EX_VA_ARG.
            if (str_eq(t.text, str_lit("va_arg")) ||
                str_eq(t.text, str_lit("__builtin_va_arg"))) {
                p->i++;  // consume '('
                Expr *e = new_expr(p, EX_VA_ARG, t.line);
                e->lhs = parse_expr(p);
                expect(p, TC_TK_COMMA, str_lit("expected ',' in va_arg(ap, T)"));
                Type ty = {0};
                if (!parse_abstract_type(p, &ty)) {
                    perror_at(p, cur(p).line,
                        str_lit("expected type in va_arg(ap, T)"));
                }
                e->cast_type = ty;
                expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
                return parse_postfix(p, e);
            }
            p->i++;
            Expr *e = new_expr(p, EX_CALL, t.line);
            e->callee = t.text;
            if (cur(p).kind != TC_TK_RPAREN) {
                for (;;) {
                    Expr *a = parse_expr(p);
                    VecExprPtr_push_back(p->arena, &e->args, a);
                    if (!accept(p, TC_TK_COMMA)) break;
                }
            }
            expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
            return parse_postfix(p, e);
        }
        Expr *e = new_expr(p, EX_VAR, t.line);
        e->name = t.text;
        return parse_postfix(p, e);
    }
    perror_at(p, t.line, str_lit("expected expression"));
    if (cur(p).kind != TC_TK_EOF) p->i++;
    Expr *e = new_expr(p, EX_INT, t.line);
    return e;
}

static Expr *parse_mul(P *p) {
    Expr *l = parse_primary(p);
    for (;;) {
        TcTokKind k = cur(p).kind;
        BinOp op;
        if      (k == TC_TK_STAR)    op = OP_MUL;
        else if (k == TC_TK_SLASH)   op = OP_DIV;
        else if (k == TC_TK_PERCENT) op = OP_MOD;
        else break;
        int line = cur(p).line;
        p->i++;
        Expr *r = parse_primary(p);
        Expr *e = new_expr(p, EX_BIN, line);
        e->op = op; e->lhs = l; e->rhs = r;
        l = e;
    }
    return l;
}
static Expr *parse_add(P *p) {
    Expr *l = parse_mul(p);
    for (;;) {
        TcTokKind k = cur(p).kind;
        BinOp op;
        if      (k == TC_TK_PLUS)  op = OP_ADD;
        else if (k == TC_TK_MINUS) op = OP_SUB;
        else break;
        int line = cur(p).line;
        p->i++;
        Expr *r = parse_mul(p);
        Expr *e = new_expr(p, EX_BIN, line);
        e->op = op; e->lhs = l; e->rhs = r;
        l = e;
    }
    return l;
}
static Expr *parse_shift(P *p) {
    Expr *l = parse_add(p);
    for (;;) {
        TcTokKind k = cur(p).kind;
        BinOp op;
        if      (k == TC_TK_SHL) op = OP_SHL;
        else if (k == TC_TK_SHR) op = OP_SHR;
        else break;
        int line = cur(p).line;
        p->i++;
        Expr *r = parse_add(p);
        Expr *e = new_expr(p, EX_BIN, line);
        e->op = op; e->lhs = l; e->rhs = r;
        l = e;
    }
    return l;
}
static Expr *parse_rel(P *p) {
    Expr *l = parse_shift(p);
    for (;;) {
        TcTokKind k = cur(p).kind;
        BinOp op;
        if      (k == TC_TK_LT)   op = OP_LT;
        else if (k == TC_TK_LE)   op = OP_LE;
        else if (k == TC_TK_GT)   op = OP_GT;
        else if (k == TC_TK_GE)   op = OP_GE;
        else break;
        int line = cur(p).line;
        p->i++;
        Expr *r = parse_shift(p);
        Expr *e = new_expr(p, EX_BIN, line);
        e->op = op; e->lhs = l; e->rhs = r;
        l = e;
    }
    return l;
}
static Expr *parse_eq(P *p) {
    Expr *l = parse_rel(p);
    for (;;) {
        TcTokKind k = cur(p).kind;
        BinOp op;
        if      (k == TC_TK_EQEQ) op = OP_EQ;
        else if (k == TC_TK_NE)   op = OP_NE;
        else break;
        int line = cur(p).line;
        p->i++;
        Expr *r = parse_rel(p);
        Expr *e = new_expr(p, EX_BIN, line);
        e->op = op; e->lhs = l; e->rhs = r;
        l = e;
    }
    return l;
}
static Expr *parse_bitand(P *p) {
    Expr *l = parse_eq(p);
    while (cur(p).kind == TC_TK_AMP) {
        int line = cur(p).line;
        p->i++;
        Expr *r = parse_eq(p);
        Expr *e = new_expr(p, EX_BIN, line);
        e->op = OP_BAND; e->lhs = l; e->rhs = r;
        l = e;
    }
    return l;
}
static Expr *parse_bitxor(P *p) {
    Expr *l = parse_bitand(p);
    while (cur(p).kind == TC_TK_CARET) {
        int line = cur(p).line;
        p->i++;
        Expr *r = parse_bitand(p);
        Expr *e = new_expr(p, EX_BIN, line);
        e->op = OP_BXOR; e->lhs = l; e->rhs = r;
        l = e;
    }
    return l;
}
static Expr *parse_bitor(P *p) {
    Expr *l = parse_bitxor(p);
    while (cur(p).kind == TC_TK_PIPE) {
        int line = cur(p).line;
        p->i++;
        Expr *r = parse_bitxor(p);
        Expr *e = new_expr(p, EX_BIN, line);
        e->op = OP_BOR; e->lhs = l; e->rhs = r;
        l = e;
    }
    return l;
}
static Expr *parse_and(P *p) {
    Expr *l = parse_bitor(p);
    while (cur(p).kind == TC_TK_AMPAMP) {
        int line = cur(p).line;
        p->i++;
        Expr *r = parse_bitor(p);
        Expr *e = new_expr(p, EX_BIN, line);
        e->op = OP_AND; e->lhs = l; e->rhs = r;
        l = e;
    }
    return l;
}
static Expr *parse_or(P *p) {
    Expr *l = parse_and(p);
    while (cur(p).kind == TC_TK_PIPEPIPE) {
        int line = cur(p).line;
        p->i++;
        Expr *r = parse_and(p);
        Expr *e = new_expr(p, EX_BIN, line);
        e->op = OP_OR; e->lhs = l; e->rhs = r;
        l = e;
    }
    return l;
}
static Expr *parse_assign_or_or(P *p);
static Expr *parse_ternary(P *p) {
    Expr *cond = parse_or(p);
    if (cur(p).kind == TC_TK_QUESTION) {
        int line = cur(p).line;
        p->i++;
        Expr *th = parse_expr(p);
        expect(p, TC_TK_COLON, str_lit("expected ':' in ternary"));
        Expr *el = parse_assign_or_or(p);  // right-associative
        Expr *e = new_expr(p, EX_TERNARY, line);
        e->lhs = cond; e->rhs = th; e->lvalue = el;
        return e;
    }
    return cond;
}
static Expr *parse_assign_or_or(P *p) {
    // Parse a logical-or / ternary expression first, then if '=' (or any
    // compound-assign) follows, treat the LHS as an lvalue (validated in
    // the emitter).
    Expr *lhs = parse_ternary(p);
    TcTokKind k = cur(p).kind;
    BinOp compound;
    bool is_compound = false;
    switch (k) {
        case TC_TK_PLUSEQ:    compound = OP_ADD;  is_compound = true; break;
        case TC_TK_MINUSEQ:   compound = OP_SUB;  is_compound = true; break;
        case TC_TK_STAREQ:    compound = OP_MUL;  is_compound = true; break;
        case TC_TK_SLASHEQ:   compound = OP_DIV;  is_compound = true; break;
        case TC_TK_PERCENTEQ: compound = OP_MOD;  is_compound = true; break;
        case TC_TK_AMPEQ:     compound = OP_BAND; is_compound = true; break;
        case TC_TK_PIPEEQ:    compound = OP_BOR;  is_compound = true; break;
        case TC_TK_CARETEQ:   compound = OP_BXOR; is_compound = true; break;
        case TC_TK_SHLEQ:     compound = OP_SHL;  is_compound = true; break;
        case TC_TK_SHREQ:     compound = OP_SHR;  is_compound = true; break;
        default: break;
    }
    if (is_compound) {
        int line = cur(p).line;
        p->i++;
        Expr *rhs = parse_assign_or_or(p);
        Expr *bin = new_expr(p, EX_BIN, line);
        bin->op = compound; bin->lhs = lhs; bin->rhs = rhs;
        Expr *e = new_expr(p, EX_ASSIGN, line);
        e->lvalue = lhs; e->rhs_assign = bin;
        return e;
    }
    if (cur(p).kind == TC_TK_ASSIGN) {
        int line = cur(p).line;
        p->i++;
        Expr *rhs = parse_assign_or_or(p);   // right-assoc
        Expr *e = new_expr(p, EX_ASSIGN, line);
        e->lvalue = lhs;
        e->rhs_assign = rhs;
        return e;
    }
    return lhs;
}

static Expr *parse_expr(P *p) { return parse_assign_or_or(p); }

// Parse a base type keyword. Returns true if we consumed one and writes
// the kind to *out. Pointer/array suffixes are handled at decl time.
//
// `out_is_long` (out) is set to true when the type was declared with the
// `long` keyword (one or more), false otherwise. Callers that build a
// `Type` should copy this into `t.is_long` so that `_Generic` can
// disambiguate `int` from `long` on wasm32 where both have storage
// kind TY_I32 / int_bits == 32.
//
// `out_is_unsigned` (out) is set to true when the type was declared
// with the `unsigned` keyword. Callers must propagate this into
// `Type.int_unsigned` so that i32 -> i64 widening uses `arith.extui`
// (zero-extend) instead of `arith.extsi` (sign-extend).
static bool parse_base_type2(P *p, TypeKind *out, bool *out_was_char,
                             bool *out_is_long, bool *out_is_unsigned) {
    *out_was_char = false;
    *out_is_long = false;
    *out_is_unsigned = false;
    skip_const(p);
    // C-style integer base specifier: any sequence of signed / unsigned /
    // short / long / int / char / _Bool / bool. `long long` is always
    // 64-bit; a single `long` is 64-bit on the native host but 32-bit
    // on wasm32 (matching clang's wasm32-wasi ABI). Everything else is
    // 32-bit.
    bool saw_int_kw = false;
    int  n_long_kw = 0;
    bool saw_signedness_kw = false;
    bool saw_unsigned_kw = false;
    bool saw_short_kw = false;
    bool saw_char_kw = false;
    bool saw_bool_kw = false;
    bool progress = true;
    while (progress) {
        progress = false;
        while (cur(p).kind == TC_TK_KW_SIGNED || cur(p).kind == TC_TK_KW_UNSIGNED) {
            saw_signedness_kw = true;
            if (cur(p).kind == TC_TK_KW_UNSIGNED) saw_unsigned_kw = true;
            p->i++; skip_const(p); progress = true;
        }
        while (cur(p).kind == TC_TK_KW_LONG) {
            n_long_kw++; p->i++; skip_const(p); progress = true;
        }
        while (cur(p).kind == TC_TK_KW_SHORT) {
            saw_short_kw = true; p->i++; skip_const(p); progress = true;
        }
        if (cur(p).kind == TC_TK_KW_INT) {
            saw_int_kw = true; p->i++; skip_const(p); progress = true;
        }
        if (cur(p).kind == TC_TK_KW_CHAR) {
            saw_char_kw = true; p->i++; skip_const(p); progress = true;
        }
        if (cur(p).kind == TC_TK_KW_BOOL) {
            saw_bool_kw = true; p->i++; skip_const(p); progress = true;
        }
    }
    *out_was_char = saw_char_kw;
    *out_is_long = (n_long_kw >= 1);
    *out_is_unsigned = saw_unsigned_kw;
    if (n_long_kw > 0) {
        bool is_64 = (n_long_kw >= 2) || !p->target_wasm32;
        *out = is_64 ? TY_I64 : TY_I32;
        skip_const(p);
        return true;
    }
    if (saw_int_kw || saw_signedness_kw || saw_short_kw || saw_char_kw || saw_bool_kw) {
        *out = TY_I32; skip_const(p); return true;
    }
    if (cur(p).kind == TC_TK_KW_FLOAT) { p->i++; *out = TY_F32; skip_const(p); return true; }
    if (cur(p).kind == TC_TK_KW_DOUBLE) { p->i++; *out = TY_F64; skip_const(p); return true; }
    if (cur(p).kind == TC_TK_KW_VOID)  { p->i++; *out = TY_VOID; skip_const(p); return true; }
    if (cur(p).kind == TC_TK_KW_ENUM) {
        // `enum [Tag]` as a type-spec — behaves exactly as `int`. A body
        // is not allowed in this position; only the top-level / statement
        // form `enum Tag { ... };` (handled separately) registers values.
        p->i++;
        if (cur(p).kind == TC_TK_IDENT) p->i++;  // optional tag
        *out = TY_I32;
        skip_const(p);
        return true;
    }
    return false;
}

static bool parse_base_type(P *p, TypeKind *out) {
    bool dummy = false;
    bool dummy_is_long = false;
    bool dummy_is_unsigned = false;
    return parse_base_type2(p, out, &dummy, &dummy_is_long, &dummy_is_unsigned);
}

static Stmt *parse_stmt(P *p);

static void parse_block(P *p, VecStmtPtr *out) {
    // Support both brace blocks `{ ... }` and unbraced single statements
    // (used everywhere in real C as the body of `if`, `else`, `while`,
    // `for`). The single-statement form only allows ONE statement.
    if (cur(p).kind != TC_TK_LBRACE) {
        Stmt *s = parse_stmt(p);
        VecStmtPtr_push_back(p->arena, out, s);
        return;
    }
    expect(p, TC_TK_LBRACE, str_lit("expected '{'"));
    while (cur(p).kind != TC_TK_RBRACE && cur(p).kind != TC_TK_EOF) {
        Stmt *s = parse_stmt(p);
        VecStmtPtr_push_back(p->arena, out, s);
    }
    expect(p, TC_TK_RBRACE, str_lit("expected '}'"));
}

// Promote a function-local `static` declaration to a module-scope
// global so the variable's value persists across calls. The original
// local name is preserved on the Stmt (so subsequent references within
// the function still bind to it via the local scope), but the Stmt is
// tagged with the mangled global name so the emitter substitutes the
// global's address for an `alloca` slot.
//
// We only promote scalars with literal initializers — the patterns
// actually used by tinyc (counters, init flags, sentinel pointers).
// Non-scalar statics (`static const char *arr[] = {...};` /
// `static struct T eof = {...};`) are left as regular locals: their
// initializers are constant data, so reinitializing on each call is
// observationally identical to a true static. The compiler emits an
// MRE-detectable error if a non-scalar `static` is genuinely mutated
// across calls (none in tinyc's own source).
static void promote_static_local(P *p, Stmt *ds, int line) {
    if (ds->kind != ST_DECL) return;
    if (p->cur_func_name.size == 0) return;  // file-scope: handled elsewhere
    Type ty = ds->decl_type;
    bool is_scalar = (ty.kind == TY_I32 || ty.kind == TY_I64 ||
                      ty.kind == TY_F32 || ty.kind == TY_F64 ||
                      ty.kind == TY_PTR_CHAR || ty.kind == TY_PTR_I32 ||
                      ty.kind == TY_PTR_VOID || ty.kind == TY_PTR_PTR ||
                      ty.kind == TY_PTR_STRUCT || ty.kind == TY_FNPTR);
    if (!is_scalar) {
        // Leave as a regular local. Non-scalar statics in tinyc's own
        // source are all const-initialized read-only tables; on every
        // call the local re-initializes to the same data, which is
        // observationally identical to a true static for these uses.
        return;
    }
    Global g = (Global){0};
    int sid = ++p->static_local_counter;
    g.name = format(p->arena,
                    str_lit("__static.{}.{}.{}"),
                    p->cur_func_name, ds->decl_name, (int64_t)sid);
    g.type = ty;
    g.line = line;
    g.is_static = true;
    if (ds->decl_init) {
        Expr *e = ds->decl_init;
        // Unwrap `(T)0` / `(T*)0` / `(void*)0` casts written by NULL-like
        // macros around an integer-zero literal so the result is treated
        // as a null pointer initializer.
        while (e && e->kind == EX_CAST && e->lhs) {
            e = e->lhs;
        }
        if (e && e->kind == EX_NULL) {
            g.has_init = true;
            g.init_int = 0;
        } else if (e && e->kind == EX_INT) {
            g.has_init = true;
            g.init_int = e->int_value;
        } else if (e && e->kind == EX_FLOAT) {
            g.has_init = true;
            g.init_float = e->float_value;
        } else if (e && e->kind == EX_STR && ty.kind == TY_PTR_CHAR) {
            g.has_init = true;
            g.init_str = e->name;
        } else {
            perror_at(p, line,
                str_lit("function-local 'static' scalar init must be a literal"));
            return;
        }
    }
    VecGlobal_push_back(p->arena, &p->prog->globals, g);
    ds->decl_static_global_sym = g.name;
    ds->decl_init = NULL;  // emitter must not re-emit the init at the use site
}

// Walk a parse_decl result (single ST_DECL or a no-scope ST_BLOCK of
// ST_DECLs) and promote every declarator to a static global.
static void promote_static_locals(P *p, Stmt *s, int line) {
    if (!s) return;
    if (s->kind == ST_DECL) {
        promote_static_local(p, s, line);
    } else if (s->kind == ST_BLOCK) {
        for (size_t i = 0; i < s->block_body.size; i++) {
            promote_static_local(p, s->block_body.data[i], line);
        }
    }
}

// Parse a decl statement: <type> [*] name [ '[' N ']' [ '[' M ']' ] ] [ '=' expr ] ';'
//   or                  : struct Name [*] name [ '[' N ']' ] [ '=' &<var> ] ';'
//   or                  : <typedef-name> name [ ... ] [ '=' expr ] ';'
// Caller has already verified that current token starts a decl.
static Stmt *parse_decl(P *p, bool require_semi) {
    int line = cur(p).line;
    bool saw_static = false;
    // Detect a leading `static` storage-class qualifier before the type
    // qualifiers so we can promote the declaration to a module-scope
    // global (function-local static semantics). `skip_const` below still
    // accepts `static` (and `inline`), but only this prefix position is
    // promoted; `static` appearing later in the qualifier list is
    // silently ignored as before.
    {
        size_t save = p->i;
        while (cur(p).kind == TC_TK_KW_CONST ||
               cur(p).kind == TC_TK_KW_STATIC ||
               cur(p).kind == TC_TK_KW_INLINE) {
            if (cur(p).kind == TC_TK_KW_STATIC) saw_static = true;
            p->i++;
        }
        p->i = save;
    }
    skip_const(p);
    // Typedef'd type-name takes precedence over manual parsing.
    if (cur(p).kind == TC_TK_IDENT && typedef_lookup(p, cur(p).text)) {
        Type ty = {0};
        parse_sig_type(p, &ty);
        TcTok name = cur(p);
        expect(p, TC_TK_IDENT, str_lit("expected identifier"));
        Stmt *s = new_stmt(p, ST_DECL, line);
        s->decl_name = name.text;
        s->decl_type = ty;
        if (accept(p, TC_TK_LBRACK)) {
            int64_t alen = 0; Expr *aexpr = NULL;
            parse_array_len_bracket(p, &alen, &aexpr);
            if (s->decl_type.kind == TY_STRUCT) {
                s->decl_type.kind = TY_ARRAY_STRUCT;
                s->decl_type.array_len = alen;
                s->decl_type.array_len_expr = aexpr;
            } else if (s->decl_type.kind == TY_PTR_I32 ||
                       s->decl_type.kind == TY_PTR_CHAR ||
                       s->decl_type.kind == TY_PTR_VOID ||
                       s->decl_type.kind == TY_PTR_STRUCT) {
                // T *arr[N] (typedef'd): collapse to TY_ARRAY_PTR_CHAR
                // since every pointer type stores as the same 8-byte
                // !llvm.ptr regardless of element kind. The element
                // type info is intentionally erased here; downstream
                // uses (loads, stores, passing the array name as a
                // function argument) all operate on the raw pointer.
                s->decl_type = (Type){0};
                s->decl_type.kind = TY_ARRAY_PTR_CHAR;
                s->decl_type.array_len = alen;
                s->decl_type.array_len_expr = aexpr;
            } else {
                bool is_i64 = (s->decl_type.kind == TY_I64);
                bool is_i8  = (s->decl_type.kind == TY_I32 &&
                               s->decl_type.int_bits == 8);
                if (s->decl_type.kind != TY_I32 && s->decl_type.kind != TY_I64) {
                    perror_at(p, line, str_lit("only int[N] / long long[N] arrays are supported"));
                }
                s->decl_type.kind = TY_ARRAY_I32;
                s->decl_type.array_elem_is_i64 = is_i64;
                s->decl_type.array_elem_is_i8  = is_i8;
                s->decl_type.array_len = alen;
                s->decl_type.array_len_expr = aexpr;
                if (accept(p, TC_TK_LBRACK)) {
                    int64_t alen2 = 0; Expr *aexpr2 = NULL;
                    parse_array_len_bracket(p, &alen2, &aexpr2);
                    s->decl_type.array_len2 = alen2;
                    if (aexpr2) perror_at(p, line, str_lit("2D array second dim must be a literal"));
                }
            }
        }
        if (accept(p, TC_TK_ASSIGN)) {
            { Expr *agg = parse_aggregate_init(p, s->decl_type); s->decl_init = agg ? agg : parse_expr(p); }
        }
        // Comma-separated additional declarators sharing the same typedef'd type.
        if (cur(p).kind == TC_TK_COMMA) {
            Stmt *blk = new_stmt(p, ST_BLOCK, line);
            blk->block_no_scope = true;
            VecStmtPtr_push_back(p->arena, &blk->block_body, s);
            while (accept(p, TC_TK_COMMA)) {
                // Accept (and ignore) any leading `*`s that match the
                // pointer-ness already absorbed into `ty` by parse_sig_type.
                // This lets `Td *a, *b;` and `Td *a, **b;` declare a/b with
                // the same pointer type as `Td*` / `Td**` without choking
                // on the per-declarator `*` tokens.
                while (cur(p).kind == TC_TK_STAR) p->i++;
                TcTok dname = cur(p);
                expect(p, TC_TK_IDENT, str_lit("expected identifier"));
                Stmt *ds = new_stmt(p, ST_DECL, dname.line);
                ds->decl_name = dname.text;
                ds->decl_type = ty;
                // Optional `[N]` (or `[N][M]`) suffix on this declarator
                // — promote the per-declarator type into the matching
                // array kind, mirroring the first-declarator branch
                // above.
                if (accept(p, TC_TK_LBRACK)) {
                    int64_t alen = 0; Expr *aexpr = NULL;
                    parse_array_len_bracket(p, &alen, &aexpr);
                    if (ds->decl_type.kind == TY_STRUCT) {
                        ds->decl_type.kind = TY_ARRAY_STRUCT;
                        ds->decl_type.array_len = alen;
                        ds->decl_type.array_len_expr = aexpr;
                    } else if (ds->decl_type.kind == TY_PTR_I32 ||
                               ds->decl_type.kind == TY_PTR_CHAR ||
                               ds->decl_type.kind == TY_PTR_VOID ||
                               ds->decl_type.kind == TY_PTR_STRUCT) {
                        ds->decl_type = (Type){0};
                        ds->decl_type.kind = TY_ARRAY_PTR_CHAR;
                        ds->decl_type.array_len = alen;
                        ds->decl_type.array_len_expr = aexpr;
                    } else {
                        bool is_i64 = (ds->decl_type.kind == TY_I64);
                        bool is_i8  = (ds->decl_type.kind == TY_I32 &&
                                       ds->decl_type.int_bits == 8);
                        if (ds->decl_type.kind != TY_I32 && ds->decl_type.kind != TY_I64) {
                            perror_at(p, line, str_lit("only int[N] / long long[N] arrays are supported"));
                        }
                        ds->decl_type.kind = TY_ARRAY_I32;
                        ds->decl_type.array_elem_is_i64 = is_i64;
                        ds->decl_type.array_elem_is_i8  = is_i8;
                        ds->decl_type.array_len = alen;
                        ds->decl_type.array_len_expr = aexpr;
                        if (accept(p, TC_TK_LBRACK)) {
                            int64_t alen2 = 0; Expr *aexpr2 = NULL;
                            parse_array_len_bracket(p, &alen2, &aexpr2);
                            ds->decl_type.array_len2 = alen2;
                            if (aexpr2) perror_at(p, line, str_lit("2D array second dim must be a literal"));
                        }
                    }
                }
                if (accept(p, TC_TK_ASSIGN)) {
                    Expr *agg = parse_aggregate_init(p, ds->decl_type);
                    ds->decl_init = agg ? agg : parse_expr(p);
                }
                VecStmtPtr_push_back(p->arena, &blk->block_body, ds);
            }
            if (require_semi) expect(p, TC_TK_SEMI, str_lit("expected ';'"));
            if (saw_static) promote_static_locals(p, blk, line);
            return blk;
        }
        if (require_semi) expect(p, TC_TK_SEMI, str_lit("expected ';'"));
        if (saw_static) promote_static_locals(p, s, line);
        return s;
    }
    // struct decl: `struct Name var;` or `struct Name * p = &s;`
    //              or `struct Name var[N];`
    //              or `struct [Name] { fields... } var [...];` (inline def)
    if (cur(p).kind == TC_TK_KW_STRUCT || cur(p).kind == TC_TK_KW_UNION) {
        // Inline struct/union body as the declaration type:
        //   struct { fields... } name [...];        — anonymous
        //   struct Tag { fields... } name [...];    — tagged-and-defined here
        bool inline_def = (peek(p, 1).kind == TC_TK_LBRACE) ||
                          (peek(p, 1).kind == TC_TK_IDENT &&
                           peek(p, 2).kind == TC_TK_LBRACE);
        TcTok sn;
        if (inline_def) {
            StructDef *sd = parse_struct_def(p);
            if (sd->name.size == 0) {
                // Synthesize an anonymous tag so the existing TY_STRUCT
                // / TY_ARRAY_STRUCT plumbing has a name to look up.
                static int local_anon_counter = 0;
                local_anon_counter++;
                char *buf = arena_new_array(p->arena, char, 32);
                int n = 0; int v = local_anon_counter;
                char digits[16]; int dn = 0;
                if (v == 0) digits[dn++] = '0';
                while (v > 0) { digits[dn++] = (char)('0' + (v % 10)); v /= 10; }
                const char *prefix = "__anon_local_";
                for (size_t k = 0; prefix[k]; k++) buf[n++] = prefix[k];
                while (dn > 0) buf[n++] = digits[--dn];
                sd->name = (string){.str = buf, .size = (size_t)n};
            }
            // Register in the program-level struct table (skipping
            // duplicates by tag name).
            bool already = false;
            for (size_t i = 0; i < p->prog->structs.size; i++) {
                if (str_eq(p->prog->structs.data[i]->name, sd->name)) {
                    already = true; break;
                }
            }
            if (!already) {
                VecStructDefPtr_push_back(p->arena, &p->prog->structs, sd);
            }
            sn = (TcTok){0};
            sn.kind = TC_TK_IDENT;
            sn.text = sd->name;
            sn.line = line;
        } else {
            p->i++;
            sn = cur(p);
            expect(p, TC_TK_IDENT, str_lit("expected struct name"));
        }
        bool is_ptr = accept(p, TC_TK_STAR);
        bool is_ptr_ptr = is_ptr && accept(p, TC_TK_STAR);
        TcTok name = cur(p);
        expect(p, TC_TK_IDENT, str_lit("expected variable name"));
        Stmt *s = new_stmt(p, ST_DECL, line);
        s->decl_name = name.text;
        s->decl_type.struct_name = sn.text;
        if (accept(p, TC_TK_LBRACK)) {
            if (is_ptr) {
                perror_at(p, line, str_lit("array of struct pointer is not supported"));
            }
            int64_t alen = 0; Expr *aexpr = NULL;
            parse_array_len_bracket(p, &alen, &aexpr);
            s->decl_type.kind = TY_ARRAY_STRUCT;
            s->decl_type.array_len = alen;
            s->decl_type.array_len_expr = aexpr;
        } else if (is_ptr_ptr) {
            // struct Foo **var: TY_PTR_PTR(pointee=TY_PTR_STRUCT).
            Type *inner = arena_new(p->arena, Type);
            *inner = (Type){0};
            inner->kind = TY_PTR_STRUCT;
            inner->struct_name = sn.text;
            s->decl_type.kind = TY_PTR_PTR;
            s->decl_type.pointee = inner;
            s->decl_type.struct_name = sn.text;
        } else {
            s->decl_type.kind = is_ptr ? TY_PTR_STRUCT : TY_STRUCT;
        }
        if (accept(p, TC_TK_ASSIGN)) {
            if (s->decl_type.kind != TY_PTR_STRUCT &&
                s->decl_type.kind != TY_PTR_PTR &&
                s->decl_type.kind != TY_STRUCT &&
                s->decl_type.kind != TY_ARRAY_STRUCT) {
                perror_at(p, line, str_lit("struct/array initializers are not supported"));
            }
            { Expr *agg = parse_aggregate_init(p, s->decl_type); s->decl_init = agg ? agg : parse_expr(p); }
        }
        if (require_semi) expect(p, TC_TK_SEMI, str_lit("expected ';'"));
        if (saw_static) promote_static_locals(p, s, line);
        return s;
    }
    // `va_list ap;` — special atomic type: must declare a single named
    // variable, no pointer/array decoration.
    if (cur(p).kind == TC_TK_KW_VA_LIST) {
        p->i++;
        TcTok name = cur(p);
        expect(p, TC_TK_IDENT, str_lit("expected identifier"));
        Stmt *s = new_stmt(p, ST_DECL, line);
        s->decl_name = name.text;
        s->decl_type.kind = TY_VA_LIST;
        if (require_semi) expect(p, TC_TK_SEMI, str_lit("expected ';'"));
        if (saw_static) promote_static_locals(p, s, line);
        return s;
    }
    TypeKind base = TY_I32;
    bool was_char = false;
    bool is_long = false;
    bool is_unsigned = false;
    parse_base_type2(p, &base, &was_char, &is_long, &is_unsigned);
    // Function-pointer local: `int (*name)(types)`.
    if (cur(p).kind == TC_TK_LPAREN && peek(p, 1).kind == TC_TK_STAR) {
        Stmt *s = new_stmt(p, ST_DECL, line);
        s->decl_type.kind = base;
        if (is_long) s->decl_type.int_bits = 64;
        s->decl_type.int_unsigned = is_unsigned;
        string nm = (string){0};
        try_parse_fnptr_suffix(p, &s->decl_type, &nm);
        if (nm.size == 0) {
            perror_at(p, line, str_lit("function-pointer declaration requires a name"));
        }
        s->decl_name = nm;
        if (accept(p, TC_TK_ASSIGN)) {
            { Expr *agg = parse_aggregate_init(p, s->decl_type); s->decl_init = agg ? agg : parse_expr(p); }
        }
        if (require_semi) expect(p, TC_TK_SEMI, str_lit("expected ';'"));
        if (saw_static) promote_static_locals(p, s, line);
        return s;
    }
    bool is_ptr = false;
    bool is_ptr_ptr = false;
    if (accept(p, TC_TK_STAR)) {
        is_ptr = true; skip_const(p);
        if (accept(p, TC_TK_STAR)) { is_ptr_ptr = true; skip_const(p); }
    }
    TcTok name = cur(p);
    expect(p, TC_TK_IDENT, str_lit("expected identifier"));
    Stmt *s = new_stmt(p, ST_DECL, line);
    s->decl_name = name.text;
    s->decl_type.kind = base;
    if (is_long) s->decl_type.int_bits = 64;
    if (!is_ptr) s->decl_type.int_unsigned = is_unsigned;
    if (is_ptr) {
        if (was_char) s->decl_type.kind = TY_PTR_CHAR;
        else if (base == TY_I32) s->decl_type.kind = TY_PTR_I32;
        else if (base == TY_I64) {
            s->decl_type.kind = TY_PTR_I32;
            s->decl_type.ptr_is_i64 = true;
        }
        else if (base == TY_F32) {
            s->decl_type.kind = TY_PTR_I32;
            s->decl_type.ptr_is_f32 = true;
        }
        else if (base == TY_F64) {
            s->decl_type.kind = TY_PTR_I32;
            s->decl_type.ptr_is_f64 = true;
        }
        else if (base == TY_VOID) s->decl_type.kind = TY_PTR_VOID;
        else perror_at(p, line, str_lit("only int*/char*/void* pointers are supported"));
        if (is_ptr_ptr) {
            // Wrap into TY_PTR_PTR carrying the inner single-level pointer.
            Type *inner = arena_new(p->arena, Type);
            *inner = s->decl_type;
            s->decl_type = (Type){0};
            s->decl_type.kind = TY_PTR_PTR;
            s->decl_type.pointee = inner;
        }
    } else if (base == TY_VOID) {
        perror_at(p, line, str_lit("'void' is not a valid variable type (did you mean 'void*'?)"));
    }
    if (accept(p, TC_TK_LBRACK)) {
        if ((base != TY_I32 && !was_char) || is_ptr_ptr) perror_at(p, line, str_lit("only int[N]/char[N]/char*[N] arrays are supported"));
        bool is_char_ptr_arr = (was_char && is_ptr);
        int64_t alen = 0; Expr *aexpr = NULL;
        parse_array_len_bracket(p, &alen, &aexpr);
        if (is_char_ptr_arr) {
            s->decl_type.kind = TY_ARRAY_PTR_CHAR;
        } else {
            s->decl_type.kind = TY_ARRAY_I32;
            if (was_char) s->decl_type.array_elem_is_i8 = true;
        }
        s->decl_type.array_len = alen;
        s->decl_type.array_len_expr = aexpr;
        if (accept(p, TC_TK_LBRACK)) {
            int64_t alen2 = 0; Expr *aexpr2 = NULL;
            parse_array_len_bracket(p, &alen2, &aexpr2);
            s->decl_type.array_len2 = alen2;
            if (aexpr2) perror_at(p, line, str_lit("2D array second dim must be a literal"));
        }
    }
    if (accept(p, TC_TK_ASSIGN)) {
        { Expr *agg = parse_aggregate_init(p, s->decl_type); s->decl_init = agg ? agg : parse_expr(p); }
        // `char arr[] = "string"` — infer array length from the string
        // literal (size includes the trailing NUL set by the lexer).
        if (s->decl_init && s->decl_init->kind == EX_STR &&
            s->decl_type.kind == TY_ARRAY_I32 &&
            s->decl_type.array_elem_is_i8 &&
            s->decl_type.array_len == 0) {
            s->decl_type.array_len = (int64_t)s->decl_init->name.size;
        }
    }
    // Comma-separated additional declarators sharing the same base type:
    //   `int i = 1, j = 2, *p = 0;`. Wrap into an ST_BLOCK if present.
    if (cur(p).kind == TC_TK_COMMA) {
        Stmt *blk = new_stmt(p, ST_BLOCK, line);
        blk->block_no_scope = true;
        VecStmtPtr_push_back(p->arena, &blk->block_body, s);
        while (accept(p, TC_TK_COMMA)) {
            bool dis_ptr = false, dis_pp = false;
            if (accept(p, TC_TK_STAR)) {
                dis_ptr = true; skip_const(p);
                if (accept(p, TC_TK_STAR)) { dis_pp = true; skip_const(p); }
            }
            TcTok dname = cur(p);
            expect(p, TC_TK_IDENT, str_lit("expected identifier"));
            Stmt *ds = new_stmt(p, ST_DECL, dname.line);
            ds->decl_name = dname.text;
            ds->decl_type.kind = base;
            if (dis_ptr) {
                if (was_char) ds->decl_type.kind = TY_PTR_CHAR;
                else if (base == TY_I32) ds->decl_type.kind = TY_PTR_I32;
                else if (base == TY_I64) {
                    ds->decl_type.kind = TY_PTR_I32;
                    ds->decl_type.ptr_is_i64 = true;
                }
                else if (base == TY_VOID) ds->decl_type.kind = TY_PTR_VOID;
                else perror_at(p, dname.line, str_lit("only int*/char*/void* pointers are supported"));
                if (dis_pp) {
                    Type *inner = arena_new(p->arena, Type);
                    *inner = ds->decl_type;
                    ds->decl_type = (Type){0};
                    ds->decl_type.kind = TY_PTR_PTR;
                    ds->decl_type.pointee = inner;
                }
            }
            if (accept(p, TC_TK_LBRACK)) {
                if ((base != TY_I32 && !was_char) || dis_pp) perror_at(p, dname.line, str_lit("only int[N]/char[N]/char*[N] arrays are supported"));
                bool dis_char_ptr_arr = (was_char && dis_ptr);
                int64_t alen = 0; Expr *aexpr = NULL;
                parse_array_len_bracket(p, &alen, &aexpr);
                ds->decl_type.kind = dis_char_ptr_arr ? TY_ARRAY_PTR_CHAR : TY_ARRAY_I32;
                ds->decl_type.array_len = alen;
                ds->decl_type.array_len_expr = aexpr;
            }
            if (accept(p, TC_TK_ASSIGN)) {
                Expr *agg = parse_aggregate_init(p, ds->decl_type);
                ds->decl_init = agg ? agg : parse_expr(p);
                if (ds->decl_init && ds->decl_init->kind == EX_STR &&
                    ds->decl_type.kind == TY_ARRAY_I32 &&
                    ds->decl_type.array_elem_is_i8 &&
                    ds->decl_type.array_len == 0) {
                    ds->decl_type.array_len = (int64_t)ds->decl_init->name.size;
                }
            }
            VecStmtPtr_push_back(p->arena, &blk->block_body, ds);
        }
        if (require_semi) expect(p, TC_TK_SEMI, str_lit("expected ';'"));
        if (saw_static) promote_static_locals(p, blk, line);
        return blk;
    }
    if (require_semi) expect(p, TC_TK_SEMI, str_lit("expected ';'"));
    if (saw_static) promote_static_locals(p, s, line);
    return s;
}

static Stmt *parse_stmt(P *p) {
    TcTok t = cur(p);
    if (t.kind == TC_TK_SEMI) {
        // Empty statement `;` — represented as an empty block.
        p->i++;
        Stmt *s = new_stmt(p, ST_BLOCK, t.line);
        return s;
    }
    if (t.kind == TC_TK_KW_TYPEDEF) {
        // Block-local typedef: register and emit an empty statement.
        parse_typedef_decl(p);
        Stmt *s = new_stmt(p, ST_BLOCK, t.line);
        return s;
    }
    if (t.kind == TC_TK_LBRACE) {
        Stmt *s = new_stmt(p, ST_BLOCK, t.line);
        parse_block(p, &s->block_body);
        return s;
    }
    if (t.kind == TC_TK_KW_GOTO) {
        p->i++;
        TcTok nm = cur(p);
        expect(p, TC_TK_IDENT, str_lit("expected label after 'goto'"));
        Stmt *s = new_stmt(p, ST_GOTO, t.line);
        s->label_name = nm.text;
        expect(p, TC_TK_SEMI, str_lit("expected ';' after goto"));
        return s;
    }
    // Plain `IDENT:` is a label statement (no nested code per stmt; the
    // following statements form the label's body in the surrounding block).
    if (t.kind == TC_TK_IDENT && peek(p, 1).kind == TC_TK_COLON) {
        p->i += 2;  // consume IDENT and ':'
        Stmt *s = new_stmt(p, ST_LABEL, t.line);
        s->label_name = t.text;
        return s;
    }
    if (t.kind == TC_TK_KW_RETURN) {
        p->i++;
        Stmt *s = new_stmt(p, ST_RETURN, t.line);
        if (cur(p).kind == TC_TK_SEMI) {
            // bare `return;` — only legal in void-returning functions
            // (validated at emit time).
            s->expr = NULL;
        } else {
            s->expr = parse_expr(p);
        }
        expect(p, TC_TK_SEMI, str_lit("expected ';' after return"));
        return s;
    }
    if (t.kind == TC_TK_KW_BREAK) {
        p->i++;
        Stmt *s = new_stmt(p, ST_BREAK, t.line);
        expect(p, TC_TK_SEMI, str_lit("expected ';' after break"));
        return s;
    }
    if (t.kind == TC_TK_KW_CONTINUE) {
        p->i++;
        Stmt *s = new_stmt(p, ST_CONTINUE, t.line);
        expect(p, TC_TK_SEMI, str_lit("expected ';' after continue"));
        return s;
    }
    if (t.kind == TC_TK_KW_IF) {
        p->i++;
        Stmt *s = new_stmt(p, ST_IF, t.line);
        expect(p, TC_TK_LPAREN, str_lit("expected '('"));
        s->cond = parse_expr(p);
        expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
        parse_block(p, &s->then_body);
        if (accept(p, TC_TK_KW_ELSE)) {
            s->has_else = true;
            parse_block(p, &s->else_body);
        }
        return s;
    }
    if (t.kind == TC_TK_KW_WHILE) {
        p->i++;
        Stmt *s = new_stmt(p, ST_WHILE, t.line);
        expect(p, TC_TK_LPAREN, str_lit("expected '('"));
        s->cond = parse_expr(p);
        expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
        parse_block(p, &s->while_body);
        return s;
    }
    if (t.kind == TC_TK_KW_DO) {
        p->i++;
        Stmt *s = new_stmt(p, ST_DO_WHILE, t.line);
        parse_block(p, &s->while_body);
        expect(p, TC_TK_KW_WHILE, str_lit("expected 'while' after do-block"));
        expect(p, TC_TK_LPAREN, str_lit("expected '('"));
        s->cond = parse_expr(p);
        expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
        expect(p, TC_TK_SEMI, str_lit("expected ';' after do-while"));
        return s;
    }
    if (t.kind == TC_TK_KW_FOR) {
        p->i++;
        Stmt *s = new_stmt(p, ST_FOR, t.line);
        expect(p, TC_TK_LPAREN, str_lit("expected '('"));
        // init: empty | decl | expr
        if (cur(p).kind == TC_TK_SEMI) {
            p->i++;
            s->for_init = NULL;
        } else if (cur(p).kind == TC_TK_KW_INT || cur(p).kind == TC_TK_KW_FLOAT || cur(p).kind == TC_TK_KW_DOUBLE ||
                   cur(p).kind == TC_TK_KW_CHAR || cur(p).kind == TC_TK_KW_ENUM ||
                   cur(p).kind == TC_TK_KW_CONST || cur(p).kind == TC_TK_KW_VOID ||
                   cur(p).kind == TC_TK_KW_STATIC || cur(p).kind == TC_TK_KW_INLINE ||
                   cur(p).kind == TC_TK_KW_LONG || cur(p).kind == TC_TK_KW_SIGNED ||
                   cur(p).kind == TC_TK_KW_UNSIGNED ||
                   cur(p).kind == TC_TK_KW_SHORT || cur(p).kind == TC_TK_KW_BOOL ||
                   (cur(p).kind == TC_TK_IDENT && typedef_lookup(p, cur(p).text))) {
            s->for_init = parse_decl(p, /*require_semi*/ true);
        } else {
            Stmt *es = new_stmt(p, ST_EXPR, cur(p).line);
            es->expr = parse_expr(p);
            expect(p, TC_TK_SEMI, str_lit("expected ';'"));
            s->for_init = es;
        }
        // cond: empty means "true"
        if (cur(p).kind != TC_TK_SEMI) s->cond = parse_expr(p);
        expect(p, TC_TK_SEMI, str_lit("expected ';' after for-cond"));
        // step: empty allowed. Multiple comma-separated expressions are
        // collected and emitted in order. C's comma operator: usually
        // `++i, ++j`.
        if (cur(p).kind != TC_TK_RPAREN) {
            VecExprPtr_push_back(p->arena, &s->for_steps, parse_expr(p));
            while (accept(p, TC_TK_COMMA)) {
                VecExprPtr_push_back(p->arena, &s->for_steps, parse_expr(p));
            }
        }
        expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
        parse_block(p, &s->for_body);
        return s;
    }
    if (t.kind == TC_TK_KW_PRINT) {
        p->i++;
        Stmt *s = new_stmt(p, ST_PRINT, t.line);
        expect(p, TC_TK_LPAREN, str_lit("expected '('"));
        s->expr = parse_expr(p);
        expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
        expect(p, TC_TK_SEMI, str_lit("expected ';'"));
        return s;
    }
    if (t.kind == TC_TK_KW_SWITCH) {
        p->i++;
        Stmt *s = new_stmt(p, ST_SWITCH, t.line);
        expect(p, TC_TK_LPAREN, str_lit("expected '('"));
        s->cond = parse_expr(p);
        expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
        expect(p, TC_TK_LBRACE, str_lit("expected '{'"));
        VecSwitchCase_reserve(p->arena, &s->switch_cases, 4);
        while (cur(p).kind != TC_TK_RBRACE && cur(p).kind != TC_TK_EOF) {
            if (cur(p).kind == TC_TK_KW_CASE) {
                p->i++;
                bool neg = false;
                if (accept(p, TC_TK_MINUS)) neg = true;
                TcTok lit = cur(p);
                int64_t cval = 0;
                if (lit.kind == TC_TK_INT_LIT) {
                    cval = lit.int_value;
                    p->i++;
                } else if (lit.kind == TC_TK_IDENT && enum_lookup(p, lit.text)) {
                    cval = enum_lookup(p, lit.text)->value;
                    p->i++;
                } else {
                    expect(p, TC_TK_INT_LIT, str_lit("expected integer or enumerator in 'case'"));
                }
                expect(p, TC_TK_COLON, str_lit("expected ':' after case"));
                SwitchCase sc = (SwitchCase){
                    .value = neg ? -cval : cval,
                    .is_default = false,
                    .body_start = s->block_body.size,
                };
                VecSwitchCase_push_back(p->arena, &s->switch_cases, sc);
                continue;
            }
            if (cur(p).kind == TC_TK_KW_DEFAULT) {
                p->i++;
                expect(p, TC_TK_COLON, str_lit("expected ':' after default"));
                SwitchCase sc = (SwitchCase){
                    .value = 0, .is_default = true,
                    .body_start = s->block_body.size,
                };
                VecSwitchCase_push_back(p->arena, &s->switch_cases, sc);
                continue;
            }
            Stmt *bs = parse_stmt(p);
            VecStmtPtr_push_back(p->arena, &s->block_body, bs);
        }
        expect(p, TC_TK_RBRACE, str_lit("expected '}'"));
        return s;
    }
    if (t.kind == TC_TK_KW_INT || t.kind == TC_TK_KW_FLOAT || t.kind == TC_TK_KW_DOUBLE ||
        t.kind == TC_TK_KW_STRUCT || t.kind == TC_TK_KW_UNION || t.kind == TC_TK_KW_CHAR ||
        t.kind == TC_TK_KW_ENUM || t.kind == TC_TK_KW_CONST ||
        t.kind == TC_TK_KW_VOID ||
        t.kind == TC_TK_KW_STATIC || t.kind == TC_TK_KW_INLINE ||
        t.kind == TC_TK_KW_LONG || t.kind == TC_TK_KW_SIGNED ||
        t.kind == TC_TK_KW_UNSIGNED ||
        t.kind == TC_TK_KW_SHORT || t.kind == TC_TK_KW_BOOL ||
        t.kind == TC_TK_KW_VA_LIST ||
        (t.kind == TC_TK_IDENT && typedef_lookup(p, t.text) &&
         (peek(p, 1).kind == TC_TK_IDENT || peek(p, 1).kind == TC_TK_STAR))) {
        // `enum [Tag] { ... };` is a module-scope-only registration form;
        // a body inside a function or block scope would leak enumerators
        // out of their lexical scope (we have no scope-pop machinery for
        // enums), so reject it. `enum [Tag] name;` (no body) still falls
        // through to parse_decl below, which treats `enum [Tag]` as `int`.
        if (t.kind == TC_TK_KW_ENUM) {
            size_t k = 1;
            if (peek(p, k).kind == TC_TK_IDENT) k++;
            if (peek(p, k).kind == TC_TK_LBRACE) {
                perror_at(p, t.line, str_lit(
                    "enum body is only allowed at module scope"));
                // Skip to matching '}' then optional ';' to keep parsing.
                while (cur(p).kind != TC_TK_LBRACE &&
                       cur(p).kind != TC_TK_EOF) p->i++;
                if (cur(p).kind == TC_TK_LBRACE) {
                    int depth = 0;
                    do {
                        if (cur(p).kind == TC_TK_LBRACE) depth++;
                        else if (cur(p).kind == TC_TK_RBRACE) depth--;
                        p->i++;
                    } while (depth > 0 && cur(p).kind != TC_TK_EOF);
                }
                accept(p, TC_TK_SEMI);
                Stmt *empty = new_stmt(p, ST_BLOCK, t.line);
                return empty;
            }
        }
        return parse_decl(p, /*require_semi*/ true);
    }
    // expression statement (covers assignments and calls)
    Stmt *s = new_stmt(p, ST_EXPR, t.line);
    s->expr = parse_expr(p);
    expect(p, TC_TK_SEMI, str_lit("expected ';'"));
    return s;
}

static bool parse_sig_type(P *p, Type *out);
static void wrap_ptr_to_ptr(P *p, Type *out);

// After a base type has been parsed, optionally consume a function-pointer
// declarator suffix: `(*[name])(T1, T2, ...)` or `(*[name])()`.
// Wraps `*ty` into a TY_FNPTR whose return type is the original `*ty` and
// parameter types are parsed from the comma list. If `out_name` is non-NULL
// and a name token is present in `(*name)`, it is stored into *out_name.
// Returns true iff the suffix was consumed.
static bool try_parse_fnptr_suffix(P *p, Type *ty, string *out_name) {
    if (cur(p).kind != TC_TK_LPAREN) return false;
    if (peek(p, 1).kind != TC_TK_STAR) return false;
    // Consume `(`, `*`.
    p->i++;
    p->i++;
    // Optional second `*` for pointer-to-function-pointer: `int (**fp)(...)`.
    bool ptr_to_fnptr = false;
    if (cur(p).kind == TC_TK_STAR) {
        ptr_to_fnptr = true;
        p->i++;
    }
    if (cur(p).kind == TC_TK_IDENT) {
        if (out_name) *out_name = cur(p).text;
        p->i++;
    }
    expect(p, TC_TK_RPAREN, str_lit("expected ')' in function-pointer declarator"));
    expect(p, TC_TK_LPAREN, str_lit("expected '(' before function-pointer parameter list"));
    Type *params = NULL;
    int nparams = 0;
    int cap = 0;
    if (cur(p).kind != TC_TK_RPAREN) {
        for (;;) {
            Type pty = {0};
            if (!parse_sig_type(p, &pty)) {
                perror_at(p, cur(p).line, str_lit("expected parameter type"));
                break;
            }
            // Allow nested fn-ptr in fn-ptr param positions.
            string dummy = (string){0};
            try_parse_fnptr_suffix(p, &pty, &dummy);
            // Skip an optional parameter name: `int (*f)(int x, int y)`.
            if (cur(p).kind == TC_TK_IDENT) p->i++;
            if (nparams == cap) {
                int ncap = cap ? cap * 2 : 4;
                Type *np = arena_new_array(p->arena, Type, (size_t)ncap);
                for (int k = 0; k < nparams; k++) np[k] = params[k];
                params = np;
                cap = ncap;
            }
            params[nparams++] = pty;
            if (!accept(p, TC_TK_COMMA)) break;
        }
    }
    expect(p, TC_TK_RPAREN, str_lit("expected ')' after function-pointer parameter list"));
    Type *ret_box = arena_new(p->arena, Type);
    *ret_box = *ty;
    *ty = (Type){0};
    ty->kind = TY_FNPTR;
    ty->fnptr_ret = ret_box;
    ty->fnptr_params = params;
    ty->fnptr_nparams = nparams;
    if (ptr_to_fnptr) {
        // Wrap as TY_PTR_PTR with the fnptr as pointee.
        wrap_ptr_to_ptr(p, ty);
    }
    return true;
}

// Wrap an already-parsed single-level pointer type into TY_PTR_PTR
// (consuming a SECOND `*`). Caller has just successfully parsed `T*` into
// *out and seen another `*` token. Limit: depth 2; we do NOT recurse to
// T*** (any further '*' is left for the caller to handle / reject).
static void wrap_ptr_to_ptr(P *p, Type *out) {
    Type *inner = arena_new(p->arena, Type);
    *inner = *out;
    *out = (Type){0};
    out->kind = TY_PTR_PTR;
    out->pointee = inner;
}

// Parse a function-signature type:  int | float | char [*] | struct Name [*]
// or a typedef'd type-name (in which case any leading '*' for typedef-int
// produces TY_PTR_I32, etc.). Also accepts a SECOND '*' to produce
// TY_PTR_PTR (e.g. `int **`, `void **`, `struct Foo **`, `BinOp *` where
// BinOp is a typedef'd function-pointer).
// Returns false if the current tokens don't start a type.
static bool parse_sig_type(P *p, Type *out) {
    *out = (Type){0};
    skip_const(p);
    // Accept signed/unsigned/short/long/char/_Bool/bool/int modifiers in any
    // order. Any 'long' promotes to TY_I64; otherwise TY_I32. If `char` is
    // among the consumed modifiers and a single trailing '*' follows, the
    // result is TY_PTR_CHAR rather than TY_PTR_I32 (matches existing tinyC
    // behaviour for `char*`).
    if (cur(p).kind == TC_TK_KW_SIGNED || cur(p).kind == TC_TK_KW_UNSIGNED ||
        cur(p).kind == TC_TK_KW_LONG   || cur(p).kind == TC_TK_KW_SHORT  ||
        cur(p).kind == TC_TK_KW_BOOL) {
        int  n_long = 0;
        bool saw_short = false;
        bool saw_char = false;
        bool saw_bool = false;
        bool saw_unsigned = false;
        while (cur(p).kind == TC_TK_KW_SIGNED || cur(p).kind == TC_TK_KW_UNSIGNED ||
               cur(p).kind == TC_TK_KW_LONG   || cur(p).kind == TC_TK_KW_INT     ||
               cur(p).kind == TC_TK_KW_SHORT  || cur(p).kind == TC_TK_KW_CHAR    ||
               cur(p).kind == TC_TK_KW_BOOL) {
            if (cur(p).kind == TC_TK_KW_LONG) n_long++;
            if (cur(p).kind == TC_TK_KW_SHORT) saw_short = true;
            if (cur(p).kind == TC_TK_KW_CHAR) saw_char = true;
            if (cur(p).kind == TC_TK_KW_BOOL) saw_bool = true;
            if (cur(p).kind == TC_TK_KW_UNSIGNED) saw_unsigned = true;
            p->i++; skip_const(p);
        }
        // `long long` is always 64-bit storage; a single `long` is
        // 64-bit storage on the native host but 32-bit on wasm32
        // (matching clang's wasm32-wasi ABI). We tag the type with
        // int_bits = 64 in BOTH cases so that `_Generic` can keep
        // `long` distinct from `int` even on wasm32 where the
        // storage kind collides (both TY_I32). The storage width is
        // tracked by `kind` (TY_I32 / TY_I64), independent of this
        // semantic tag.
        bool long_storage_64 = (n_long >= 2) || (n_long == 1 && !p->target_wasm32);
        out->kind = long_storage_64 ? TY_I64 : TY_I32;
        if (n_long >= 1) out->int_bits = 64;
        else if (saw_char) out->int_bits = 8;
        else if (saw_short) out->int_bits = 16;
        else if (saw_bool) out->int_bits = 8;
        else out->int_bits = 32;
        out->int_unsigned = saw_unsigned;
        skip_const(p);
        if (accept(p, TC_TK_STAR)) {
            // `long *` → pointer to long. Storage element width matches
            // the long storage width: i64 on native, i32 on wasm32. We
            // model the i64 element via TY_PTR_I32 with ptr_is_i64.
            out->kind = saw_char ? TY_PTR_CHAR : TY_PTR_I32;
            if (long_storage_64 && !saw_char) out->ptr_is_i64 = true;
            skip_const(p);
            if (accept(p, TC_TK_STAR)) { wrap_ptr_to_ptr(p, out); skip_const(p); }
        }
        return true;
    }
    if (cur(p).kind == TC_TK_KW_INT)   {
        p->i++; out->kind = TY_I32; out->int_bits = 32; skip_const(p);
        if (accept(p, TC_TK_STAR)) {
            out->kind = TY_PTR_I32;
            skip_const(p);
            if (accept(p, TC_TK_STAR)) { wrap_ptr_to_ptr(p, out); skip_const(p); }
        }
        skip_const(p);
        return true;
    }
    if (cur(p).kind == TC_TK_KW_FLOAT) {
        p->i++; out->kind = TY_F32; skip_const(p);
        if (accept(p, TC_TK_STAR)) {
            out->kind = TY_PTR_I32; out->ptr_is_f32 = true;
            skip_const(p);
            if (accept(p, TC_TK_STAR)) { wrap_ptr_to_ptr(p, out); skip_const(p); }
        }
        skip_const(p);
        return true;
    }
    if (cur(p).kind == TC_TK_KW_DOUBLE) {
        p->i++; out->kind = TY_F64; skip_const(p);
        if (accept(p, TC_TK_STAR)) {
            out->kind = TY_PTR_I32; out->ptr_is_f64 = true;
            skip_const(p);
            if (accept(p, TC_TK_STAR)) { wrap_ptr_to_ptr(p, out); skip_const(p); }
        }
        skip_const(p);
        return true;
    }
    if (cur(p).kind == TC_TK_KW_VOID)  {
        p->i++;
        skip_const(p);
        if (accept(p, TC_TK_STAR)) {
            out->kind = TY_PTR_VOID;
            skip_const(p);
            if (accept(p, TC_TK_STAR)) { wrap_ptr_to_ptr(p, out); skip_const(p); }
        } else {
            out->kind = TY_VOID;
        }
        skip_const(p);
        return true;
    }
    if (cur(p).kind == TC_TK_KW_CHAR)  {
        p->i++;
        skip_const(p);
        if (accept(p, TC_TK_STAR)) {
            out->kind = TY_PTR_CHAR;
            skip_const(p);
            if (accept(p, TC_TK_STAR)) { wrap_ptr_to_ptr(p, out); skip_const(p); }
        } else {
            out->kind = TY_I32;
            out->int_bits = 8;
        }
        skip_const(p);
        return true;
    }
    if (cur(p).kind == TC_TK_KW_STRUCT || cur(p).kind == TC_TK_KW_UNION) {
        p->i++;
        TcTok sn = cur(p);
        if (!expect(p, TC_TK_IDENT, str_lit("expected struct name"))) return false;
        out->kind = TY_STRUCT;
        out->struct_name = sn.text;
        skip_const(p);
        if (accept(p, TC_TK_STAR)) {
            out->kind = TY_PTR_STRUCT;
            skip_const(p);
            if (accept(p, TC_TK_STAR)) { wrap_ptr_to_ptr(p, out); skip_const(p); }
        }
        skip_const(p);
        return true;
    }
    if (cur(p).kind == TC_TK_KW_ENUM) {
        // `enum [Tag]` — a parser-side alias for `int`. No body allowed in
        // a signature/cast position; the top-level `enum Tag { ... };`
        // form is handled by the program-level loop.
        p->i++;
        if (cur(p).kind == TC_TK_IDENT) p->i++;
        out->kind = TY_I32;
        skip_const(p);
        if (accept(p, TC_TK_STAR)) {
            out->kind = TY_PTR_I32;
            skip_const(p);
            if (accept(p, TC_TK_STAR)) { wrap_ptr_to_ptr(p, out); skip_const(p); }
        }
        skip_const(p);
        return true;
    }
    if (cur(p).kind == TC_TK_KW_VA_LIST) {
        // tinyC's emitter only supports va_list fully as a local variable
        // declaration. Accept it in parameter / typedef position too so
        // that headers with declarations like `void f(..., va_list ap)`
        // parse cleanly (the emitter treats it as an opaque pointer-sized
        // value when it actually has to emit one).
        p->i++;
        out->kind = TY_VA_LIST;
        skip_const(p);
        return true;
    }
    if (cur(p).kind == TC_TK_IDENT) {
        Typedef *td = typedef_lookup(p, cur(p).text);
        if (td) {
            p->i++;
            *out = td->ty;
            skip_const(p);
            // Allow `Td*` to add another level of pointer (only for scalar
            // typedefs; struct typedefs already encode the pointer).
            if (accept(p, TC_TK_STAR)) {
                if (out->kind == TY_I32 && out->int_bits == 8) {
                    // `uint8_t *` / `int8_t *` / similar: 8-bit element
                    // type. Treat as `char *` so pointer arithmetic uses
                    // a 1-byte stride rather than 4.
                    out->kind = TY_PTR_CHAR;
                }
                else if (out->kind == TY_I32) out->kind = TY_PTR_I32;
                else if (out->kind == TY_I64) {
                    out->kind = TY_PTR_I32;
                    out->ptr_is_i64 = true;
                }
                else if (out->kind == TY_STRUCT) out->kind = TY_PTR_STRUCT;
                else if (out->kind == TY_FNPTR) {
                    // `BinOp *` where BinOp is a typedef'd function pointer.
                    // Wrap into TY_PTR_PTR(pointee=TY_FNPTR copy).
                    wrap_ptr_to_ptr(p, out);
                }
                // else: ignore (e.g. typedef'd pointer + extra '*' not supported)
                skip_const(p);
                if (accept(p, TC_TK_STAR) &&
                    (out->kind == TY_PTR_I32 || out->kind == TY_PTR_STRUCT ||
                     out->kind == TY_PTR_CHAR || out->kind == TY_PTR_VOID)) {
                    wrap_ptr_to_ptr(p, out);
                }
            }
            skip_const(p);
            return true;
        }
    }
    return false;
}

// Parse a base type, then optionally an abstract fn-ptr declarator
// `(*)(types)`. Used by sizeof / cast positions.
static bool parse_abstract_type(P *p, Type *out) {
    if (!parse_sig_type(p, out)) return false;
    string dummy = (string){0};
    try_parse_fnptr_suffix(p, out, &dummy);
    // Optional array suffix: `T[]` (unsized — common in compound-literal
    // positions like `(T[]){...}`) or `T[N]`. Wraps the element type into
    // a TY_ARRAY_STRUCT (struct base) or TY_ARRAY_I32 (int/char base).
    if (cur(p).kind == TC_TK_LBRACK) {
        p->i++;  // '['
        int64_t alen = 0; Expr *aexpr = NULL;
        parse_array_len_bracket(p, &alen, &aexpr);
        if (out->kind == TY_STRUCT) {
            out->kind = TY_ARRAY_STRUCT;
        } else if (out->kind == TY_PTR_CHAR) {
            out->kind = TY_ARRAY_PTR_CHAR;
        } else {
            bool is_i64 = (out->kind == TY_I64);
            bool is_i8  = (out->kind == TY_I32 && out->int_bits == 8);
            out->kind = TY_ARRAY_I32;
            out->array_elem_is_i64 = is_i64;
            out->array_elem_is_i8  = is_i8;
        }
        out->array_len = alen;
        out->array_len_expr = aexpr;
    }
    return true;
}

static Func *parse_func(P *p) {
    int line = cur(p).line;
    // Allow `__attribute__((...))` between the storage class (already
    // consumed by the caller) and the return type — this covers the
    // common `__attribute__((export_name("..."))) <type> name(...)`
    // pattern.
    AttrInfo attrs = (AttrInfo){0};
    parse_attributes(p, &attrs);
    Type ret_ty = {0};
    if (!parse_sig_type(p, &ret_ty)) {
        perror_at(p, cur(p).line, str_lit("expected return type"));
    }
    // Also between the return type and the function name (the WASI
    // macro expansion in corec/platform/platform_wasm.c produces this
    // shape after preprocessing: `uint32_t __attribute__((...)) name(...)`).
    parse_attributes(p, &attrs);
    TcTok name = cur(p);
    expect(p, TC_TK_IDENT, str_lit("expected function name"));
    expect(p, TC_TK_LPAREN, str_lit("expected '('"));
    Func *f = arena_new(p->arena, Func);
    *f = (Func){0};
    f->name = name.text;
    f->return_type = ret_ty;
    f->line = line;
    f->wasm_import_module = attrs.import_module;
    f->wasm_import_name = attrs.import_name;
    f->wasm_export_name = attrs.export_name;
    VecParam_reserve(p->arena, &f->params, 4);
    VecStmtPtr_reserve(p->arena, &f->body, 8);
    // `f(void)` — single `void` token (not `void*` or named) means no params.
    if (cur(p).kind == TC_TK_KW_VOID && peek(p, 1).kind == TC_TK_RPAREN) {
        p->i++;  // consume `void`
    } else if (cur(p).kind != TC_TK_RPAREN) {
        for (;;) {
            // `...` as a "parameter": variadic marker. Must be the last
            // entry in the list, and there must be at least one fixed
            // parameter before it (matches C99 `f(int x, ...)`).
            if (cur(p).kind == TC_TK_ELLIPSIS) {
                if (f->params.size == 0) {
                    perror_at(p, cur(p).line,
                        str_lit("'...' requires at least one named parameter"));
                }
                p->i++;
                f->is_variadic = true;
                break;
            }
            int pline = cur(p).line;
            Type pty = {0};
            if (!parse_sig_type(p, &pty)) {
                perror_at(p, cur(p).line, str_lit("expected parameter type"));
            }
            if (pty.kind == TY_VOID) {
                perror_at(p, pline,
                    str_lit("'void' is only valid as the lone parameter spec"));
            }
            string pname = (string){0};
            // Optional function-pointer parameter declarator: `int (*f)(int)`.
            if (try_parse_fnptr_suffix(p, &pty, &pname)) {
                // pname (if any) was captured inside `(*name)`.
            } else if (cur(p).kind == TC_TK_IDENT) {
                pname = cur(p).text;
                p->i++;
                // C function-parameter array decay: `T name[N]` is a synonym
                // for `T *name`. Consume one or more `[expr?]` suffixes and
                // promote the base type to the corresponding pointer kind.
                // Multi-dimensional `T name[N1][N2]` is treated as a flat
                // pointer (we don't carry the inner array shape).
                while (accept(p, TC_TK_LBRACK)) {
                    while (cur(p).kind != TC_TK_RBRACK &&
                           cur(p).kind != TC_TK_EOF) {
                        p->i++;
                    }
                    expect(p, TC_TK_RBRACK,
                           str_lit("expected ']' in parameter array suffix"));
                    if (pty.kind == TY_I32 && pty.int_bits == 8) {
                        pty.kind = TY_PTR_CHAR;
                        pty.int_bits = 0;
                    } else if (pty.kind == TY_I32) {
                        pty.kind = TY_PTR_I32;
                    } else if (pty.kind == TY_I64) {
                        pty.kind = TY_PTR_I32;
                        pty.ptr_is_i64 = true;
                    } else if (pty.kind == TY_F32) {
                        pty.kind = TY_PTR_I32;
                        pty.ptr_is_f32 = true;
                    } else if (pty.kind == TY_F64) {
                        pty.kind = TY_PTR_I32;
                        pty.ptr_is_f64 = true;
                    } else if (pty.kind == TY_STRUCT) {
                        pty.kind = TY_PTR_STRUCT;
                    }
                    // For pointer / array / fnptr base types: leave the type
                    // as-is (we'd need TY_PTR_PTR or a richer model to
                    // represent `char *argv[]`); but consuming the suffix
                    // at least lets the parser keep going.
                }
            }
            VecParam_push_back(p->arena, &f->params,
                ((Param){.name = pname, .type = pty, .line = cur(p).line}));
            if (!accept(p, TC_TK_COMMA)) break;
        }
    }
    expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
    // Trailing `__attribute__((...))` (e.g. `int foo(void) __attribute__((const));`).
    {
        AttrInfo trailing = (AttrInfo){0};
        parse_attributes(p, &trailing);
        if (trailing.import_module.size > 0) f->wasm_import_module = trailing.import_module;
        if (trailing.import_name.size > 0) f->wasm_import_name = trailing.import_name;
        if (trailing.export_name.size > 0) f->wasm_export_name = trailing.export_name;
    }
    if (accept(p, TC_TK_SEMI)) {
        // Forward declaration / prototype: no body.
        f->is_forward = true;
        return f;
    }
    string saved_func = p->cur_func_name;
    p->cur_func_name = f->name;
    parse_block(p, &f->body);
    p->cur_func_name = saved_func;
    return f;
}

static StructDef *parse_struct_def(P *p) {
    int line = cur(p).line;
    bool is_union = false;
    if (cur(p).kind == TC_TK_KW_UNION) {
        is_union = true;
        p->i++;
    } else {
        expect(p, TC_TK_KW_STRUCT, str_lit("expected 'struct'"));
    }
    string name = (string){0};
    if (cur(p).kind == TC_TK_IDENT) {
        name = cur(p).text;
        p->i++;
    }
    // Anonymous struct (no tag): caller must patch sd->name afterwards.
    expect(p, TC_TK_LBRACE, str_lit("expected '{'"));
    StructDef *sd = arena_new(p->arena, StructDef);
    *sd = (StructDef){0};
    sd->name = name;
    sd->line = line;
    sd->is_union = is_union;
    VecStructField_reserve(p->arena, &sd->fields, 4);
    while (cur(p).kind != TC_TK_RBRACE && cur(p).kind != TC_TK_EOF) {
        skip_const(p);
        Type ft = {0};
        bool is_struct_kind = false;
        bool is_ptr = false;
        bool was_char = false;
        bool was_float = false;
        if ((cur(p).kind == TC_TK_KW_STRUCT || cur(p).kind == TC_TK_KW_UNION) &&
            (peek(p, 1).kind == TC_TK_LBRACE ||
             (peek(p, 1).kind == TC_TK_IDENT && peek(p, 2).kind == TC_TK_LBRACE))) {
            // Nested anonymous (or tagged) struct/union as a field type:
            //   struct { ... } name;
            //   union  { ... } name;
            // Parse the body via parse_struct_def, synthesize a name if
            // anonymous, register it in prog->structs, and treat the field
            // as a TY_STRUCT pointing at it.
            StructDef *nested = parse_struct_def(p);
            if (nested->name.size == 0) {
                static int anon_counter = 0;
                anon_counter++;
                char *buf = arena_new_array(p->arena, char, 32);
                int n = 0; int v = anon_counter;
                char digits[16]; int dn = 0;
                if (v == 0) digits[dn++] = '0';
                while (v > 0) { digits[dn++] = (char)('0' + (v % 10)); v /= 10; }
                const char *prefix = "__anon_";
                for (size_t k = 0; prefix[k]; k++) buf[n++] = prefix[k];
                while (dn > 0) buf[n++] = digits[--dn];
                nested->name = (string){.str = buf, .size = (size_t)n};
            }
            VecStructDefPtr_push_back(p->arena, &p->prog->structs, nested);
            ft.kind = TY_STRUCT;
            ft.struct_name = nested->name;
            is_struct_kind = true;
            // Optional pointer suffix: `struct Tag { ... } *name;`
            // promotes the field type to TY_PTR_STRUCT. Used for
            // self-referential / forward-list-style fields in tinyc itself
            // (e.g. `struct LabelBlock { ...; struct LabelBlock *next; } *labels;`).
            if (accept(p, TC_TK_STAR)) {
                is_ptr = true;
                ft.kind = TY_PTR_STRUCT;
            }
        } else if (cur(p).kind == TC_TK_KW_STRUCT || cur(p).kind == TC_TK_KW_UNION) {
            // Nested struct field: `struct Inner name;` (by-value),
            // `struct Inner* name;` (pointer to struct), or
            // `struct Inner* name[N];` (array of struct pointers).
            p->i++;
            TcTok sn = cur(p);
            if (!expect(p, TC_TK_IDENT, str_lit("expected struct name"))) { p->i++; continue; }
            is_ptr = accept(p, TC_TK_STAR);
            ft.kind = is_ptr ? TY_PTR_STRUCT : TY_STRUCT;
            ft.struct_name = sn.text;
            is_struct_kind = true;
        } else if (cur(p).kind == TC_TK_KW_ENUM &&
                   (peek(p, 1).kind == TC_TK_LBRACE ||
                    (peek(p, 1).kind == TC_TK_IDENT && peek(p, 2).kind == TC_TK_LBRACE))) {
            // Anonymous (or tagged) enum body inline in struct:
            //   enum { K_A, K_B } kind;
            // Register enumerators globally (parser-level), and treat the
            // field as TY_I32.
            p->i++;
            if (cur(p).kind == TC_TK_IDENT) p->i++;
            if (cur(p).kind == TC_TK_LBRACE) parse_enum_body(p);
            ft.kind = TY_I32;
            ft.int_bits = 32;
        } else if (cur(p).kind == TC_TK_KW_ENUM) {
            // `enum Tag` field: treat as int.
            p->i++;
            if (cur(p).kind == TC_TK_IDENT) p->i++;
            ft.kind = TY_I32;
            ft.int_bits = 32;
        } else if (cur(p).kind == TC_TK_IDENT && typedef_lookup(p, cur(p).text)) {
            // Typedef'd field type, e.g. `size_t buf_len;` or `ciovec_t* p;`.
            // We resolve the typedef and reuse the existing pointer/array
            // logic by mapping the resolved type to one of the supported
            // field kinds.
            Typedef *td = typedef_lookup(p, cur(p).text);
            p->i++;
            ft = td->ty;
            // Pointer suffix `*` applies to the resolved typedef.
            if (accept(p, TC_TK_STAR)) {
                is_ptr = true;
                if (ft.kind == TY_STRUCT) {
                    ft.kind = TY_PTR_STRUCT;
                } else if (ft.kind == TY_VOID) {
                    ft.kind = TY_PTR_VOID;
                } else if (ft.kind == TY_I32 && ft.int_bits == 8) {
                    ft.kind = TY_PTR_CHAR;
                } else if (ft.kind == TY_I32 || ft.kind == TY_I64) {
                    // No distinct TY_PTR_I64 today — bucket integer
                    // pointers under TY_PTR_I32 as a conservative alias.
                    bool is_i64 = (ft.kind == TY_I64);
                    ft.kind = TY_PTR_I32;
                    ft.ptr_is_i64 = is_i64;
                } else {
                    perror_at(p, cur(p).line,
                        str_lit("unsupported pointer-to-typedef field type"));
                }
                if (accept(p, TC_TK_STAR)) {
                    Type *inner = arena_new(p->arena, Type);
                    *inner = ft;
                    ft = (Type){0};
                    ft.kind = TY_PTR_PTR;
                    ft.pointee = inner;
                }
            }
        } else {
            TypeKind k;
            was_char = false;
            bool field_is_long = false;
            bool field_is_unsigned = false;
            was_float = (cur(p).kind == TC_TK_KW_FLOAT);
            if (!parse_base_type2(p, &k, &was_char, &field_is_long, &field_is_unsigned)) {
                perror_at(p, cur(p).line, str_lit("expected field type (int|float|char|struct)"));
                p->i++;
                continue;
            }
            ft.kind = k;
            if (field_is_long) ft.int_bits = 64;
            ft.int_unsigned = field_is_unsigned;
            // Optional pointer suffix on base types: `T *` (single) or
            // `T **` (pointer-to-pointer). The `**` form wraps the
            // single-level pointer kind into a TY_PTR_PTR.
            if (accept(p, TC_TK_STAR)) {
                is_ptr = true;
                if (was_char) ft.kind = TY_PTR_CHAR;
                else if (k == TY_I32) ft.kind = TY_PTR_I32;
                else if (k == TY_I64) { ft.kind = TY_PTR_I32; ft.ptr_is_i64 = true; }
                else if (k == TY_VOID) ft.kind = TY_PTR_VOID;
                else {
                    perror_at(p, cur(p).line, str_lit("only int*/char*/void* pointer fields are supported"));
                }
                if (accept(p, TC_TK_STAR)) {
                    Type *inner = arena_new(p->arena, Type);
                    *inner = ft;
                    ft = (Type){0};
                    ft.kind = TY_PTR_PTR;
                    ft.pointee = inner;
                }
            } else if (k == TY_VOID) {
                perror_at(p, cur(p).line, str_lit("'void' is not a valid struct field type"));
            }
        }
        // One or more comma-separated declarators sharing the base type
        // `ft` parsed above. Each may add its own `[N]`/`[N][M]` suffix.
        // We also accept (and tolerate) per-declarator pointer suffixes
        // matching the one already applied to the base, e.g.
        //   const uint8_t *p, *end;
        // which is equivalent to two pointer fields. The first `*` was
        // already consumed when the base type was parsed; the second
        // `*` is redundant given how the base-type code already applies
        // the pointer to all declarators in the list. We don't yet
        // model the `int *a, b;` case where `b` is non-pointer.
        Type base_ft = ft;
        bool more_decls = true;
        while (more_decls) {
            Type ft = base_ft;
            // Consume optional `*` (or `**`) leader on this declarator.
            // The pointer kind was already applied to base_ft, so we
            // just skip the tokens.
            if (accept(p, TC_TK_STAR)) {
                (void)accept(p, TC_TK_STAR);
            }
            TcTok fn = cur(p);
            expect(p, TC_TK_IDENT, str_lit("expected field name"));
            // Optional array suffix `[N]` or `[N][M]`.
            if (accept(p, TC_TK_LBRACK)) {
                if (cur(p).kind == TC_TK_RBRACK) {
                    perror_at(p, cur(p).line, str_lit("flexible array fields are not supported"));
                }
                TcTok lit = cur(p);
                expect(p, TC_TK_INT_LIT, str_lit("expected array length"));
                expect(p, TC_TK_RBRACK, str_lit("expected ']'"));
                int64_t n1 = lit.int_value;
                int64_t n2 = 0;
                if (accept(p, TC_TK_LBRACK)) {
                    TcTok lit2 = cur(p);
                    expect(p, TC_TK_INT_LIT, str_lit("expected array length"));
                    expect(p, TC_TK_RBRACK, str_lit("expected ']'"));
                    n2 = lit2.int_value;
                }
                // Map (element-type, dim) -> array TypeKind.
                if (is_struct_kind) {
                    if (!is_ptr) {
                        perror_at(p, cur(p).line, str_lit("array of nested struct value is not supported as a field"));
                    } else if (n2 != 0) {
                        perror_at(p, cur(p).line, str_lit("only 1D arrays of struct pointers are supported"));
                    } else {
                        ft.kind = TY_ARRAY_PTR_STRUCT;
                        ft.array_len = n1;
                    }
                } else if (was_char && is_ptr) {
                    if (n2 != 0) {
                        perror_at(p, cur(p).line, str_lit("only 1D arrays of char* are supported"));
                    }
                    ft.kind = TY_ARRAY_PTR_CHAR;
                    ft.array_len = n1;
                } else if (is_ptr) {
                    perror_at(p, cur(p).line, str_lit("array of int* is not supported as a field"));
                } else if (was_float) {
                    ft.kind = TY_ARRAY_F32;
                    ft.array_len = n1;
                    ft.array_len2 = n2;
                } else {
                    bool is_i64 = (ft.kind == TY_I64);
                    bool is_i8  = (ft.kind == TY_I32 && ft.int_bits == 8);
                    ft.kind = TY_ARRAY_I32;
                    ft.array_len = n1;
                    ft.array_len2 = n2;
                    ft.array_elem_is_i64 = is_i64;
                    ft.array_elem_is_i8  = is_i8;
                }
            }
            VecStructField_push_back(p->arena, &sd->fields,
                ((StructField){.name = fn.text, .type = ft}));
            more_decls = accept(p, TC_TK_COMMA);
        }
        expect(p, TC_TK_SEMI, str_lit("expected ';' after field"));
    }
    expect(p, TC_TK_RBRACE, str_lit("expected '}'"));
    return sd;
}

// Parse `enum [Tag] { name [= int-const-expr], ... };` at module-scope or
// statement-scope. Registers each enumerator in the parser's enum map.
// The `enum` keyword has already been peeked but NOT consumed by the
// caller. The trailing `;` is consumed here.
//
// Constant expression accepted for `= ...`: an INT_LIT, optionally with
// a leading unary `-`, OR a previously-declared enumerator name. This is
// intentionally a tight subset (per the spec).
// Parse the body of an enum:  `{ NAME [= EXPR], ... }`. Caller has
// already consumed `enum` and the optional tag and verified that the
// next token is `{`. Does not consume any trailing punctuation after `}`.
static void parse_enum_body(P *p) {
    expect(p, TC_TK_LBRACE, str_lit("expected '{' to begin enum body"));
    int64_t next_value = 0;
    while (cur(p).kind != TC_TK_RBRACE && cur(p).kind != TC_TK_EOF) {
        TcTok nm = cur(p);
        if (!expect(p, TC_TK_IDENT, str_lit("expected enumerator name"))) break;
        int64_t value = next_value;
        if (accept(p, TC_TK_ASSIGN)) {
            int line = cur(p).line;
            Expr *init = parse_ternary(p);
            int64_t v;
            if (!parse_const_eval(p, init, &v)) {
                perror_at(p, line,
                    str_lit("enum initializer must be a constant "
                            "expression of int literals and previously-"
                            "declared enumerators"));
            } else {
                value = v;
            }
        }
        if (enum_lookup(p, nm.text)) {
            perror_at(p, nm.line, str_lit("duplicate enumerator"));
        }
        Enumerator *e = arena_new(p->arena, Enumerator);
        e->name = nm.text;
        e->value = value;
        e->next = p->enums;
        p->enums = e;
        next_value = value + 1;
        if (!accept(p, TC_TK_COMMA)) break;
    }
    expect(p, TC_TK_RBRACE, str_lit("expected '}'"));
}

static void parse_enum_decl_top(P *p) {
    int line = cur(p).line;
    expect(p, TC_TK_KW_ENUM, str_lit("expected 'enum'"));
    // Optional tag — accepted for source compatibility, not stored. The
    // tag namespace is implicit: `enum Tag` always lowers to `int`, so
    // there's nothing to look up later.
    if (cur(p).kind == TC_TK_IDENT) p->i++;
    parse_enum_body(p);
    expect(p, TC_TK_SEMI, str_lit("expected ';' after enum declaration"));
    (void)line;
}

Program *tinyc_parse(Arena *arena, VecTcTok toks) {
    Program *prog = arena_new(arena, Program);
    *prog = (Program){0};
    tinyc_parse_into(arena, prog, toks, /*target_wasm32=*/false);
    return prog;
}

// Compare two struct definitions structurally. Two struct definitions of
// the same name across files (e.g., included from a shared header) must
// have identical field lists.
static bool struct_def_equal(StructDef *a, StructDef *b) {
    if (a->fields.size != b->fields.size) return false;
    for (size_t i = 0; i < a->fields.size; i++) {
        StructField *fa = &a->fields.data[i];
        StructField *fb = &b->fields.data[i];
        if (!str_eq(fa->name, fb->name)) return false;
        if (fa->type.kind != fb->type.kind) return false;
        if (fa->type.array_len != fb->type.array_len) return false;
        if (!str_eq(fa->type.struct_name, fb->type.struct_name)) return false;
    }
    return true;
}

// Push a global onto `prog->globals`, applying tentative-definition
// merging across files:
//   * existing has_init=false + new has_init=false -> dedup.
//   * existing has_init=false + new has_init=true  -> replace.
//   * existing has_init=true  + new has_init=false -> drop new.
//   * both has_init=true                           -> error.
static void merge_push_global(P *p, Program *prog, Global g) {
    for (size_t i = 0; i < prog->globals.size; i++) {
        Global *ex = &prog->globals.data[i];
        if (!str_eq(ex->name, g.name)) continue;
        if (ex->type.kind != g.type.kind) {
            perror_at(p, g.line,
                str_lit("conflicting global declaration types"));
            return;
        }
        if (!ex->has_init && g.has_init) {
            *ex = g;        // tentative replaced by definition
        } else if (ex->has_init && g.has_init) {
            perror_at(p, g.line, str_lit("global redefinition"));
        }
        // tentative + tentative or definition + tentative: dedup.
        return;
    }
    VecGlobal_push_back(p->arena, &prog->globals, g);
}

// Parse a `typedef ...;` declaration. Assumes cur(p) is TC_TK_KW_TYPEDEF.
// Used at file scope and inside function bodies (block-local typedefs).
static void parse_typedef_decl(P *p) {
    Program *prog = p->prog;
    Arena *arena = p->arena;
    int line = cur(p).line;
    p->i++;
    // Skip typedefs of the va_list keyword to itself (or aliases).
    // tinyC already provides `va_list` as a built-in; corec does
    // `typedef __builtin_va_list va_list;` which we map to a no-op.
    if (cur(p).kind == TC_TK_KW_VA_LIST) {
        while (cur(p).kind != TC_TK_SEMI && cur(p).kind != TC_TK_EOF) p->i++;
        accept(p, TC_TK_SEMI);
        return;
    }
    // `typedef enum [Tag] { ... } Alias;`
    if (cur(p).kind == TC_TK_KW_ENUM) {
        p->i++;
        if (cur(p).kind == TC_TK_IDENT) p->i++;  // optional tag
        if (cur(p).kind == TC_TK_LBRACE) {
            parse_enum_body(p);
        }
        if (cur(p).kind == TC_TK_IDENT) {
            TcTok nm = cur(p);
            p->i++;
            Type ty = (Type){0};
            ty.kind = TY_I32;
            Typedef *td = arena_new(arena, Typedef);
            td->name = nm.text;
            td->ty = ty;
            td->next = p->typedefs;
            p->typedefs = td;
        }
        expect(p, TC_TK_SEMI, str_lit("expected ';' after typedef"));
        return;
    }
    // `typedef struct [Tag] { ... } Alias;` or `typedef union ...`
    if (cur(p).kind == TC_TK_KW_STRUCT || cur(p).kind == TC_TK_KW_UNION) {
        size_t la = 1;
        if (peek(p, la).kind == TC_TK_IDENT) la++;
        if (peek(p, la).kind == TC_TK_LBRACE) {
            StructDef *sd = parse_struct_def(p);
            string alias = (string){0};
            if (cur(p).kind == TC_TK_IDENT) {
                alias = cur(p).text;
                p->i++;
            }
            if (sd->name.size == 0) {
                if (alias.size == 0) {
                    perror_at(p, line,
                        str_lit("typedef of anonymous struct requires a name"));
                }
                sd->name = alias;
            }
            StructDef *existing_sd = NULL;
            for (size_t i = 0; i < prog->structs.size; i++) {
                if (str_eq(prog->structs.data[i]->name, sd->name)) {
                    existing_sd = prog->structs.data[i];
                    break;
                }
            }
            if (!existing_sd) {
                VecStructDefPtr_push_back(arena, &prog->structs, sd);
            } else if (existing_sd->fields.size == 0) {
                existing_sd->fields = sd->fields;
                existing_sd->line = sd->line;
            } else if (!struct_def_equal(existing_sd, sd)) {
                perror_at(p, sd->line, str_lit("conflicting redefinition of struct"));
            }
            if (alias.size != 0) {
                Type ty = (Type){0};
                ty.kind = TY_STRUCT;
                ty.struct_name = sd->name;
                Typedef *td = arena_new(arena, Typedef);
                td->name = alias;
                td->ty = ty;
                td->next = p->typedefs;
                p->typedefs = td;
            }
            expect(p, TC_TK_SEMI, str_lit("expected ';' after typedef"));
            return;
        }
        // No body: `typedef struct Tag Alias;`.
        if (peek(p, 1).kind == TC_TK_IDENT) {
            string tag = peek(p, 1).text;
            bool found = false;
            for (size_t i = 0; i < prog->structs.size; i++) {
                if (str_eq(prog->structs.data[i]->name, tag)) {
                    found = true; break;
                }
            }
            if (!found) {
                StructDef *sd = arena_new(arena, StructDef);
                *sd = (StructDef){0};
                sd->name = tag;
                sd->line = line;
                VecStructField_reserve(arena, &sd->fields, 0);
                VecStructDefPtr_push_back(arena, &prog->structs, sd);
            }
        }
        // Fall through to parse_sig_type below (which handles `struct Tag`).
    }
    Type ty = {0};
    if (!parse_sig_type(p, &ty)) {
        perror_at(p, line, str_lit("expected type after 'typedef'"));
        while (cur(p).kind != TC_TK_SEMI && cur(p).kind != TC_TK_EOF) p->i++;
        accept(p, TC_TK_SEMI);
        return;
    }
    // Function-pointer typedef: `typedef int (*BinOp)(int, int);`.
    string fnp_name = (string){0};
    if (try_parse_fnptr_suffix(p, &ty, &fnp_name)) {
        if (fnp_name.size == 0) {
            perror_at(p, line, str_lit("function-pointer typedef requires a name"));
        }
        expect(p, TC_TK_SEMI, str_lit("expected ';' after typedef"));
        Typedef *td = arena_new(arena, Typedef);
        td->name = fnp_name;
        td->ty = ty;
        td->next = p->typedefs;
        p->typedefs = td;
        return;
    }
    TcTok nm = cur(p);
    if (!expect(p, TC_TK_IDENT, str_lit("expected typedef name"))) return;
    // Optional `[N]` to support `typedef int Vec[N];` (1D only).
    if (accept(p, TC_TK_LBRACK)) {
        TcTok lit = cur(p);
        expect(p, TC_TK_INT_LIT, str_lit("expected array length"));
        expect(p, TC_TK_RBRACK, str_lit("expected ']'"));
        if (ty.kind == TY_I32) {
            bool is_i8 = (ty.int_bits == 8);
            ty.kind = TY_ARRAY_I32;
            ty.array_elem_is_i8 = is_i8;
            ty.array_len = lit.int_value;
        } else {
            perror_at(p, line, str_lit("typedef array only supported for int"));
        }
    }
    expect(p, TC_TK_SEMI, str_lit("expected ';' after typedef"));
    Typedef *td = arena_new(arena, Typedef);
    td->name = nm.text;
    td->ty = ty;
    td->next = p->typedefs;
    p->typedefs = td;
}

int tinyc_parse_into(Arena *arena, Program *prog, VecTcTok toks, bool target_wasm32) {
    prog->target_wasm32 = target_wasm32;
    P p = {.arena = arena, .toks = toks.data, .n = toks.size, .i = 0,
           .typedefs = NULL, .enums = NULL, .prog = prog,
           .target_wasm32 = target_wasm32};
    // Built-in typedefs. `int64_t` / `uint64_t` are always 64-bit;
    // `size_t` / `ssize_t` / `intptr_t` / `uintptr_t` / `ptrdiff_t`
    // are pointer-sized in C, which means 32-bit on wasm32 and
    // 64-bit on the native (host) targets we support.
    {
        const char *names[] = {"int64_t", "uint64_t", "size_t", "ssize_t",
                               "intptr_t", "uintptr_t", "ptrdiff_t"};
        bool unsigneds[] = {false, true, true, false, false, true, false};
        // int64_t/uint64_t are always 64-bit; the rest follow the
        // pointer/long sizing for the target.
        bool always_64[]  = {true,  true,  false, false, false, false, false};
        for (size_t k = 0; k < sizeof(names) / sizeof(names[0]); k++) {
            Typedef *td = arena_new(arena, Typedef);
            *td = (Typedef){0};
            td->name = (string){.str = (char *)names[k], .size = 0};
            // Compute string length without libc.
            const char *s = names[k]; size_t len = 0; while (s[len]) len++;
            td->name.size = len;
            bool is_64 = always_64[k] || !target_wasm32;
            td->ty.kind = is_64 ? TY_I64 : TY_I32;
            td->ty.int_bits = is_64 ? 64 : 32;
            td->ty.int_unsigned = unsigneds[k];
            td->next = p.typedefs;
            p.typedefs = td;
        }
        // int32_t / uint32_t — match TY_I32 to keep existing semantics.
        const char *n32[] = {"int32_t", "uint32_t"};
        bool u32[] = {false, true};
        for (size_t k = 0; k < sizeof(n32) / sizeof(n32[0]); k++) {
            Typedef *td = arena_new(arena, Typedef);
            *td = (Typedef){0};
            const char *s = n32[k]; size_t len = 0; while (s[len]) len++;
            td->name = (string){.str = (char *)n32[k], .size = len};
            td->ty.kind = TY_I32;
            td->ty.int_bits = 32;
            td->ty.int_unsigned = u32[k];
            td->next = p.typedefs;
            p.typedefs = td;
        }
        // int8_t / int16_t / uint8_t / uint16_t — narrow ints. Bucketed
        // to TY_I32 storage but tagged with int_bits for `_Generic`.
        const char *nn[] = {"int8_t", "uint8_t", "int16_t", "uint16_t"};
        int       nbits[] = {8, 8, 16, 16};
        bool      nuns[] = {false, true, false, true};
        for (size_t k = 0; k < sizeof(nn) / sizeof(nn[0]); k++) {
            Typedef *td = arena_new(arena, Typedef);
            *td = (Typedef){0};
            const char *s = nn[k]; size_t len = 0; while (s[len]) len++;
            td->name = (string){.str = (char *)nn[k], .size = len};
            td->ty.kind = TY_I32;
            td->ty.int_bits = nbits[k];
            td->ty.int_unsigned = nuns[k];
            td->next = p.typedefs;
            p.typedefs = td;
        }
    }
    if (prog->funcs.max == 0) {
        VecFuncPtr_reserve(arena, &prog->funcs, 4);
        VecStructDefPtr_reserve(arena, &prog->structs, 4);
        VecGlobal_reserve(arena, &prog->globals, 4);
    }
    while (cur(&p).kind != TC_TK_EOF) {
        // Top-level storage-class qualifiers. Track `static` so we can emit
        // private linkage for static functions/globals.
        bool tl_saw_static = false;
        while (cur(&p).kind == TC_TK_KW_STATIC || cur(&p).kind == TC_TK_KW_INLINE) {
            if (cur(&p).kind == TC_TK_KW_STATIC) tl_saw_static = true;
            p.i++;
        }
        // Leading `__attribute__((...))` annotations at file scope.
        // Attach the recognized wasm import/export attributes to the
        // function (or drop them silently for globals where we don't
        // model them yet).
        AttrInfo tl_attrs = (AttrInfo){0};
        parse_attributes(&p, &tl_attrs);
        if (cur(&p).kind == TC_TK_EOF) break;
        if ((cur(&p).kind == TC_TK_KW_STRUCT || cur(&p).kind == TC_TK_KW_UNION) &&
            (peek(&p, 1).kind == TC_TK_LBRACE ||
             (peek(&p, 1).kind == TC_TK_IDENT && peek(&p, 2).kind == TC_TK_LBRACE))) {
            StructDef *sd = parse_struct_def(&p);
            expect(&p, TC_TK_SEMI, str_lit("expected ';' after struct definition"));
            // Dedup against existing struct of the same name (cross-file
            // include of a shared header). Identical -> skip; mismatched
            // -> diagnostic.
            StructDef *existing_sd = NULL;
            for (size_t i = 0; i < prog->structs.size; i++) {
                if (str_eq(prog->structs.data[i]->name, sd->name)) {
                    existing_sd = prog->structs.data[i];
                    break;
                }
            }
            if (!existing_sd) {
                VecStructDefPtr_push_back(arena, &prog->structs, sd);
            } else if (existing_sd->fields.size == 0) {
                // Existing entry is a forward-declared placeholder; upgrade
                // it in place by copying the new full definition's fields.
                existing_sd->fields = sd->fields;
                existing_sd->line = sd->line;
            } else if (!struct_def_equal(existing_sd, sd)) {
                perror_at(&p, sd->line,
                    str_lit("conflicting redefinition of struct"));
            }
            continue;
        }
        // Forward declaration of an opaque struct: `struct Tag;` — no body.
        // We register a StructDef with no fields so that TY_PTR_STRUCT
        // references don't trip find_struct lookups in the emitter. If
        // the struct gains a body later, that real definition will dedup
        // the field set and overwrite the placeholder via the regular
        // struct merge path.
        if (cur(&p).kind == TC_TK_KW_STRUCT &&
            peek(&p, 1).kind == TC_TK_IDENT &&
            peek(&p, 2).kind == TC_TK_SEMI) {
            int line = cur(&p).line;
            p.i++;
            string nm = cur(&p).text;
            p.i++;  // IDENT
            p.i++;  // SEMI
            StructDef *existing = NULL;
            for (size_t i = 0; i < prog->structs.size; i++) {
                if (str_eq(prog->structs.data[i]->name, nm)) {
                    existing = prog->structs.data[i];
                    break;
                }
            }
            if (!existing) {
                StructDef *sd = arena_new(arena, StructDef);
                *sd = (StructDef){0};
                sd->name = nm;
                sd->line = line;
                VecStructField_reserve(arena, &sd->fields, 0);
                VecStructDefPtr_push_back(arena, &prog->structs, sd);
            }
            continue;
        }
        // Top-level enum body: `enum [Tag] { ... };`. Anything else that
        // starts with `enum` (a global/function whose type-spec is `enum
        // Tag`) falls through to the generic decl path below, where
        // parse_sig_type treats `enum [Tag]` as `int`.
        if (cur(&p).kind == TC_TK_KW_ENUM) {
            size_t k = 1;
            if (peek(&p, k).kind == TC_TK_IDENT) k++;
            if (peek(&p, k).kind == TC_TK_LBRACE) {
                parse_enum_decl_top(&p);
                continue;
            }
        }
        if (cur(&p).kind == TC_TK_KW_TYPEDEF) {
            parse_typedef_decl(&p);
            continue;
        }
        // `extern` on a global makes it externally linked (e.g. libc's
        // `stdout`). For function decls it's still parser-noise.
        bool saw_extern = false;
        if (cur(&p).kind == TC_TK_KW_EXTERN) { saw_extern = true; p.i++; }
        // `static` and `inline` at file scope: track so we can emit private
        // linkage for static functions/globals.
        bool saw_static = tl_saw_static;
        while (cur(&p).kind == TC_TK_KW_STATIC ||
               cur(&p).kind == TC_TK_KW_INLINE) {
            if (cur(&p).kind == TC_TK_KW_STATIC) saw_static = true;
            p.i++;
        }
        // Disambiguate top-level decl between a function and a global.
        // We look ahead past `<type>` (and optional `*`) for an IDENT
        // followed by `=` or `;` -> global, otherwise function.
        size_t save = p.i;
        Type tty = {0};
        if (parse_sig_type(&p, &tty)) {
            // Function-pointer global: `int (*name)(types);`.
            if (cur(&p).kind == TC_TK_LPAREN && peek(&p, 1).kind == TC_TK_STAR) {
                string fnp_name = (string){0};
                try_parse_fnptr_suffix(&p, &tty, &fnp_name);
                if (fnp_name.size == 0) {
                    perror_at(&p, cur(&p).line,
                              str_lit("function-pointer global requires a name"));
                }
                expect(&p, TC_TK_SEMI, str_lit("expected ';' after global function pointer"));
                Global g = (Global){0};
                g.name = fnp_name;
                g.type = tty;
                g.is_extern = saw_extern;
                g.is_static = saw_static;
                g.line = cur(&p).line;
                merge_push_global(&p, prog, g);
                continue;
            }
            if (cur(&p).kind == TC_TK_IDENT) {
                TcTokKind after = peek(&p, 1).kind;
                if (after == TC_TK_ASSIGN || after == TC_TK_SEMI ||
                    after == TC_TK_LBRACK) {
                    TcTok nm = cur(&p);
                    p.i++;
                    Global g = (Global){0};
                    g.name = nm.text;
                    g.is_extern = saw_extern;
                    g.is_static = saw_static;
                    g.type = tty;
                    g.line = nm.line;
                    // Optional array suffix `[N]` (or `[const-expr]`).
                    // Only zero-initialized arrays are supported at file
                    // scope; no aggregate initializer.
                    if (accept(&p, TC_TK_LBRACK)) {
                        int64_t alen = 0; Expr *aexpr = NULL;
                        parse_array_len_bracket(&p, &alen, &aexpr);
                        if (g.type.kind == TY_STRUCT) {
                            g.type.kind = TY_ARRAY_STRUCT;
                        } else if (g.type.kind == TY_PTR_CHAR) {
                            g.type.kind = TY_ARRAY_PTR_CHAR;
                        } else if (g.type.kind == TY_PTR_STRUCT ||
                                   g.type.kind == TY_PTR_VOID ||
                                   g.type.kind == TY_FNPTR ||
                                   g.type.kind == TY_PTR_PTR ||
                                   g.type.kind == TY_PTR_I32) {
                            g.type.kind = TY_ARRAY_PTR_STRUCT;
                        } else if (g.type.kind == TY_I32 || g.type.kind == TY_I64) {
                            bool is_i64 = (g.type.kind == TY_I64);
                            bool is_i8  = (g.type.kind == TY_I32 && g.type.int_bits == 8);
                            g.type.kind = TY_ARRAY_I32;
                            g.type.array_elem_is_i64 = is_i64;
                            g.type.array_elem_is_i8  = is_i8;
                        } else {
                            perror_at(&p, nm.line,
                                str_lit("unsupported global array element type"));
                        }
                        g.type.array_len = alen;
                        g.type.array_len_expr = aexpr;
                    }
                    if (accept(&p, TC_TK_ASSIGN)) {
                        g.has_init = true;
                        TcTok lit = cur(&p);
                        if (lit.kind == TC_TK_LBRACE) {
                            // Aggregate initializer for a global array.
                            // Each element must be an int/long literal
                            // (optionally negated), NULL, or a `(cast)0`
                            // sequence — i.e. constant-evaluable at
                            // parse time. Pack the values little-endian
                            // into init_array_data when at least one
                            // non-zero value is present; otherwise fall
                            // through to the existing zero-init path.
                            p.i++;
                            size_t elem_size = 4;
                            if (g.type.kind == TY_ARRAY_I32) {
                                elem_size = g.type.array_elem_is_i64 ? 8
                                          : g.type.array_elem_is_i8  ? 1
                                          : 4;
                            }
                            int64_t alen = g.type.array_len;
                            uint8_t *bytes = NULL;
                            size_t   bcap  = 0;
                            if (alen > 0 && (g.type.kind == TY_ARRAY_I32)) {
                                bcap  = (size_t)alen * elem_size;
                                bytes = (uint8_t *)arena_alloc(p.arena, bcap);
                                for (size_t bi = 0; bi < bcap; bi++) bytes[bi] = 0;
                            }
                            size_t  ei = 0;
                            bool    any_nonzero = false;
                            if (cur(&p).kind != TC_TK_RBRACE) {
                                for (;;) {
                                    TcTok el = cur(&p);
                                    bool ok = false;
                                    int64_t val = 0;
                                    bool have_int_val = false;
                                    if (el.kind == TC_TK_INT_LIT) {
                                        val = el.int_value;
                                        have_int_val = true;
                                        ok = true;
                                    } else if (el.kind == TC_TK_KW_NULL) {
                                        val = 0; have_int_val = true; ok = true;
                                    } else if (el.kind == TC_TK_MINUS &&
                                               peek(&p, 1).kind == TC_TK_INT_LIT) {
                                        val = -peek(&p, 1).int_value;
                                        have_int_val = true;
                                        ok = true;
                                        p.i++;       // skip the '-'
                                        el = cur(&p); // now the int lit
                                    } else if (el.kind == TC_TK_LPAREN) {
                                        // tolerate `(void*)0`-style tokens by
                                        // skipping to matching ')'. Followed
                                        // by an int 0 constant.
                                        int depth = 1; p.i++;
                                        while (depth > 0 && cur(&p).kind != TC_TK_EOF) {
                                            if (cur(&p).kind == TC_TK_LPAREN) depth++;
                                            else if (cur(&p).kind == TC_TK_RPAREN) depth--;
                                            p.i++;
                                        }
                                        // optional int after the cast
                                        if (cur(&p).kind == TC_TK_INT_LIT) {
                                            val = cur(&p).int_value;
                                            have_int_val = true;
                                            p.i++;
                                        }
                                        ok = true;
                                    }
                                    if (!ok) {
                                        perror_at(&p, el.line,
                                            str_lit("global array initializer element must be int/long literal or NULL"));
                                        p.i++;
                                    } else if (el.kind != TC_TK_LPAREN) {
                                        p.i++;
                                    }
                                    if (have_int_val && bytes && ei < (size_t)alen) {
                                        uint64_t uv = (uint64_t)val;
                                        for (size_t b = 0; b < elem_size; b++) {
                                            bytes[ei * elem_size + b] = (uint8_t)(uv >> (8 * b));
                                        }
                                        if (val != 0) any_nonzero = true;
                                    }
                                    if (have_int_val) ei++;
                                    if (cur(&p).kind == TC_TK_COMMA) {
                                        p.i++;
                                        if (cur(&p).kind == TC_TK_RBRACE) break;
                                        continue;
                                    }
                                    break;
                                }
                            }
                            expect(&p, TC_TK_RBRACE,
                                   str_lit("expected '}' in global array initializer"));
                            if (any_nonzero && bytes && bcap > 0) {
                                g.init_array_data.str  = (char *)bytes;
                                g.init_array_data.size = bcap;
                            }
                        } else if (lit.kind == TC_TK_INT_LIT &&
                                   peek(&p, 1).kind == TC_TK_SHL &&
                                   peek(&p, 2).kind == TC_TK_INT_LIT) {
                            int64_t a = lit.int_value;
                            int64_t b = peek(&p, 2).int_value;
                            g.init_int = a << b;
                            p.i += 3;
                        } else if (lit.kind == TC_TK_INT_LIT) {
                            g.init_int = lit.int_value;
                            p.i++;
                        } else if (lit.kind == TC_TK_FLOAT_LIT) {
                            g.init_float = lit.float_value;
                            p.i++;
                        } else if (lit.kind == TC_TK_KW_NULL) {
                            // a null-initialized pointer; init_int=0 means null.
                            p.i++;
                        } else if (lit.kind == TC_TK_STRING_LIT) {
                            g.init_str = lit.text;
                            p.i++;
                        } else if (lit.kind == TC_TK_MINUS &&
                                   peek(&p, 1).kind == TC_TK_INT_LIT) {
                            p.i++;
                            TcTok n = cur(&p);
                            g.init_int = -n.int_value;
                            p.i++;
                        } else if (lit.kind == TC_TK_MINUS &&
                                   peek(&p, 1).kind == TC_TK_FLOAT_LIT) {
                            p.i++;
                            TcTok n = cur(&p);
                            g.init_float = -n.float_value;
                            p.i++;
                        } else if (lit.kind == TC_TK_IDENT && enum_lookup(&p, lit.text)) {
                            g.init_int = enum_lookup(&p, lit.text)->value;
                            p.i++;
                        } else if (lit.kind == TC_TK_LPAREN) {
                            // `((void*)0)` — scalar null pointer init via cast.
                            // Also `(int32_t)<int-lit>` and similar simple
                            // single-token cast forms used in tinyc's own
                            // source (e.g. `(int32_t)0x80000000`); recognise
                            // these by looking for an INT_LIT or MINUS+INT_LIT
                            // after a single matching `)`.
                            int depth = 0;
                            while (cur(&p).kind == TC_TK_LPAREN) { depth++; p.i++; }
                            while (cur(&p).kind != TC_TK_RPAREN &&
                                   cur(&p).kind != TC_TK_EOF) p.i++;
                            if (cur(&p).kind == TC_TK_RPAREN) { p.i++; depth--; }
                            bool neg = accept(&p, TC_TK_MINUS);
                            if (cur(&p).kind == TC_TK_INT_LIT) {
                                int64_t iv = cur(&p).int_value;
                                g.init_int = neg ? -iv : iv;
                                p.i++;
                            } else if (cur(&p).kind == TC_TK_FLOAT_LIT) {
                                double fv = cur(&p).float_value;
                                g.init_float = neg ? -fv : fv;
                                p.i++;
                            } else if (neg) {
                                // Roll back the consumed MINUS — it wasn't a
                                // negative number literal.
                                p.i--;
                            }
                            while (depth > 0 && cur(&p).kind == TC_TK_RPAREN) {
                                p.i++; depth--;
                            }
                        } else if (lit.kind == TC_TK_INT_LIT &&
                                   peek(&p, 1).kind == TC_TK_SHL &&
                                   peek(&p, 2).kind == TC_TK_INT_LIT) {
                            int64_t a = lit.int_value;
                            int64_t b = peek(&p, 2).int_value;
                            g.init_int = a << b;
                            p.i += 3;
                        } else {
                            perror_at(&p, lit.line,
                                str_lit("global initializer must be a literal"));
                            p.i++;
                        }
                    }
                    expect(&p, TC_TK_SEMI, str_lit("expected ';' after global"));
                    merge_push_global(&p, prog, g);
                    continue;
                }
            }
        }
        // Not a global — rewind and parse as a function (def or forward decl).
        p.i = save;
        Func *f = parse_func(&p);
        f->is_static = saw_static;
        // Merge top-level `__attribute__((...))` annotations (parsed
        // before the storage-class / return type) into the function. If
        // both the leading and the inner attribute lists set the same
        // field, prefer the more specific (inner) one already populated.
        if (f->wasm_import_module.size == 0) f->wasm_import_module = tl_attrs.import_module;
        if (f->wasm_import_name.size == 0)   f->wasm_import_name   = tl_attrs.import_name;
        if (f->wasm_export_name.size == 0)   f->wasm_export_name   = tl_attrs.export_name;
        // Check for an existing entry with the same name. Forward decls can
        // be replaced by a definition; duplicate forward decls are merged;
        // a forward decl after a definition is dropped.
        Func *existing = NULL;
        size_t existing_idx = 0;
        for (size_t i = 0; i < prog->funcs.size; i++) {
            if (str_eq(prog->funcs.data[i]->name, f->name)) {
                existing = prog->funcs.data[i];
                existing_idx = i;
                break;
            }
        }
        if (!existing) {
            VecFuncPtr_push_back(arena, &prog->funcs, f);
        } else {
            if (existing->is_variadic != f->is_variadic) {
                perror_at(&p, f->line,
                    str_lit("variadic-ness mismatch between forward decl and definition"));
            }
            if (!existing->is_forward && f->is_forward) {
                // Drop redundant forward decl after definition.
            } else if (existing->is_forward && f->is_forward) {
                // Drop redundant forward decl after another forward decl.
            } else if (existing->is_forward && !f->is_forward) {
                // Replace the forward decl with the real definition.
                prog->funcs.data[existing_idx] = f;
            } else {
                perror_at(&p, f->line, str_lit("function redefinition"));
            }
        }
    }
    // Publish the parser's enumerator list to the Program for emit-time
    // resolution (lookup falls back to it after locals/globals/functions).
    for (Enumerator *en = p.enums; en; en = en->next) {
        ProgramEnum *pe = arena_new(arena, ProgramEnum);
        pe->name = en->name;
        pe->value = en->value;
        pe->next = prog->enums;
        prog->enums = pe;
    }
    return p.err_count;
}

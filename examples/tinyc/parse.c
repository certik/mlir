// tinyC parser — recursive descent. Produces an AST defined in tinyc.h.

#include "tinyc.h"

#include <base/arena.h>
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

static void perror_at(P *p, int line, string msg) {
    println(str_lit("tinyc parse error at line {}: {}"), (int64_t)line, msg);
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
    VecStmtPtr_reserve(p->arena, &s->block_body, 4);
    return s;
}

static Expr *parse_expr(P *p);
static bool parse_sig_type(P *p, Type *out);
static bool parse_abstract_type(P *p, Type *out);
static bool try_parse_fnptr_suffix(P *p, Type *ty, string *out_name);
static void parse_enum_decl_top(P *p);

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
            // Postfix ++/--: desugar to `base = base ± 1`. The result of
            // the expression is the new value (we don't preserve the
            // pre-increment value). Suitable for expression-statement /
            // for-step usage.
            BinOp op = (cur(p).kind == TC_TK_PLUSPLUS) ? OP_ADD : OP_SUB;
            int line = cur(p).line;
            p->i++;
            Expr *one = new_expr(p, EX_INT, line);
            one->int_value = 1;
            Expr *bin = new_expr(p, EX_BIN, line);
            bin->op = op; bin->lhs = base; bin->rhs = one;
            Expr *as = new_expr(p, EX_ASSIGN, line);
            as->lvalue = base; as->rhs_assign = bin;
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
        return e;
    }
    if (t.kind == TC_TK_FLOAT_LIT) {
        p->i++;
        Expr *e = new_expr(p, EX_FLOAT, t.line);
        e->float_value = t.float_value;
        return e;
    }
    if (t.kind == TC_TK_KW_NULL) {
        p->i++;
        Expr *e = new_expr(p, EX_NULL, t.line);
        return e;
    }
    if (t.kind == TC_TK_KW_SIZEOF) {
        p->i++;
        expect(p, TC_TK_LPAREN, str_lit("expected '(' after sizeof"));
        // Disambiguate `sizeof(<type>)` vs `sizeof(<expr>)`. A type-name
        // starts with a base-type keyword, `const`, `enum`, or a typedef
        // identifier. Anything else is parsed as an expression.
        Expr *e = new_expr(p, EX_SIZEOF, t.line);
        TcTokKind nxt = cur(p).kind;
        bool looks_like_type =
            nxt == TC_TK_KW_INT || nxt == TC_TK_KW_FLOAT ||
            nxt == TC_TK_KW_CHAR || nxt == TC_TK_KW_VOID ||
            nxt == TC_TK_KW_STRUCT || nxt == TC_TK_KW_ENUM ||
            nxt == TC_TK_KW_CONST || nxt == TC_TK_KW_LONG ||
            nxt == TC_TK_KW_SIGNED || nxt == TC_TK_KW_UNSIGNED ||
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
        Expr *e = new_expr(p, EX_STR, t.line);
        e->name = t.text;
        return e;
    }
    if (t.kind == TC_TK_LPAREN) {
        // Could be a cast `(T)expr` or a parenthesized expression.
        TcTokKind nxt = peek(p, 1).kind;
        bool is_cast = (nxt == TC_TK_KW_INT || nxt == TC_TK_KW_FLOAT ||
                        nxt == TC_TK_KW_STRUCT || nxt == TC_TK_KW_CHAR ||
                        nxt == TC_TK_KW_CONST || nxt == TC_TK_KW_LONG ||
                        nxt == TC_TK_KW_SIGNED || nxt == TC_TK_KW_UNSIGNED);
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
                // Compound literal: (T){ v0, v1, ... }
                p->i++;  // consume '{'
                Expr *e = new_expr(p, EX_COMPOUND, t.line);
                e->cast_type = ty;
                if (cur(p).kind != TC_TK_RBRACE) {
                    for (;;) {
                        Expr *v = parse_expr(p);
                        VecExprPtr_push_back(p->arena, &e->args, v);
                        if (cur(p).kind == TC_TK_COMMA) { p->i++; continue; }
                        break;
                    }
                }
                expect(p, TC_TK_RBRACE,
                       str_lit("expected '}' in compound literal"));
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
    if (t.kind == TC_TK_IDENT) {
        p->i++;
        if (cur(p).kind == TC_TK_LPAREN) {
            // Built-in `va_arg(ap, T)`: second argument is a TYPE-NAME, not
            // an expression. Lower to EX_VA_ARG.
            if (str_eq(t.text, str_lit("va_arg"))) {
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
static bool parse_base_type(P *p, TypeKind *out) {
    skip_const(p);
    // C-style integer base specifier: optional signed/unsigned, optional
    // long [long], optional int. Any 'long' makes the result TY_I64.
    bool saw_int_kw = false;
    bool saw_long_kw = false;
    bool saw_signedness_kw = false;
    while (cur(p).kind == TC_TK_KW_SIGNED || cur(p).kind == TC_TK_KW_UNSIGNED) {
        saw_signedness_kw = true; p->i++; skip_const(p);
    }
    while (cur(p).kind == TC_TK_KW_LONG) {
        saw_long_kw = true; p->i++; skip_const(p);
    }
    if (cur(p).kind == TC_TK_KW_INT) {
        saw_int_kw = true; p->i++; skip_const(p);
    }
    while (cur(p).kind == TC_TK_KW_LONG) {
        saw_long_kw = true; p->i++; skip_const(p);
    }
    if (cur(p).kind == TC_TK_KW_SIGNED || cur(p).kind == TC_TK_KW_UNSIGNED) {
        saw_signedness_kw = true; p->i++; skip_const(p);
    }
    if (saw_long_kw)         { *out = TY_I64; skip_const(p); return true; }
    if (saw_int_kw || saw_signedness_kw) {
        *out = TY_I32; skip_const(p); return true;
    }
    if (cur(p).kind == TC_TK_KW_FLOAT) { p->i++; *out = TY_F32; skip_const(p); return true; }
    if (cur(p).kind == TC_TK_KW_CHAR)  { p->i++; *out = TY_I32; skip_const(p); return true; }
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

static Stmt *parse_stmt(P *p);

static void parse_block(P *p, VecStmtPtr *out) {
    expect(p, TC_TK_LBRACE, str_lit("expected '{'"));
    while (cur(p).kind != TC_TK_RBRACE && cur(p).kind != TC_TK_EOF) {
        Stmt *s = parse_stmt(p);
        VecStmtPtr_push_back(p->arena, out, s);
    }
    expect(p, TC_TK_RBRACE, str_lit("expected '}'"));
}

// Parse a decl statement: <type> [*] name [ '[' N ']' [ '[' M ']' ] ] [ '=' expr ] ';'
//   or                  : struct Name [*] name [ '[' N ']' ] [ '=' &<var> ] ';'
//   or                  : <typedef-name> name [ ... ] [ '=' expr ] ';'
// Caller has already verified that current token starts a decl.
static Stmt *parse_decl(P *p, bool require_semi) {
    int line = cur(p).line;
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
            TcTok lit = cur(p);
            expect(p, TC_TK_INT_LIT, str_lit("expected array length"));
            expect(p, TC_TK_RBRACK, str_lit("expected ']'"));
            if (s->decl_type.kind != TY_I32) {
                perror_at(p, line, str_lit("only int[N] arrays are supported"));
            }
            s->decl_type.kind = TY_ARRAY_I32;
            s->decl_type.array_len = lit.int_value;
            if (accept(p, TC_TK_LBRACK)) {
                TcTok lit2 = cur(p);
                expect(p, TC_TK_INT_LIT, str_lit("expected array length"));
                expect(p, TC_TK_RBRACK, str_lit("expected ']'"));
                s->decl_type.array_len2 = lit2.int_value;
            }
        }
        if (accept(p, TC_TK_ASSIGN)) {
            s->decl_init = parse_expr(p);
        }
        if (require_semi) expect(p, TC_TK_SEMI, str_lit("expected ';'"));
        return s;
    }
    // struct decl: `struct Name var;` or `struct Name * p = &s;`
    //              or `struct Name var[N];`
    if (cur(p).kind == TC_TK_KW_STRUCT) {
        p->i++;
        TcTok sn = cur(p);
        expect(p, TC_TK_IDENT, str_lit("expected struct name"));
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
            TcTok lit = cur(p);
            expect(p, TC_TK_INT_LIT, str_lit("expected array length"));
            expect(p, TC_TK_RBRACK, str_lit("expected ']'"));
            s->decl_type.kind = TY_ARRAY_STRUCT;
            s->decl_type.array_len = lit.int_value;
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
                s->decl_type.kind != TY_STRUCT) {
                perror_at(p, line, str_lit("struct/array initializers are not supported"));
            }
            s->decl_init = parse_expr(p);
        }
        if (require_semi) expect(p, TC_TK_SEMI, str_lit("expected ';'"));
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
        return s;
    }
    TypeKind base = TY_I32;
    bool was_char = (cur(p).kind == TC_TK_KW_CHAR);
    parse_base_type(p, &base);
    // Function-pointer local: `int (*name)(types)`.
    if (cur(p).kind == TC_TK_LPAREN && peek(p, 1).kind == TC_TK_STAR) {
        Stmt *s = new_stmt(p, ST_DECL, line);
        s->decl_type.kind = base;
        string nm = (string){0};
        try_parse_fnptr_suffix(p, &s->decl_type, &nm);
        if (nm.size == 0) {
            perror_at(p, line, str_lit("function-pointer declaration requires a name"));
        }
        s->decl_name = nm;
        if (accept(p, TC_TK_ASSIGN)) {
            s->decl_init = parse_expr(p);
        }
        if (require_semi) expect(p, TC_TK_SEMI, str_lit("expected ';'"));
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
    if (is_ptr) {
        if (was_char) s->decl_type.kind = TY_PTR_CHAR;
        else if (base == TY_I32) s->decl_type.kind = TY_PTR_I32;
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
        if (base != TY_I32 || is_ptr) perror_at(p, line, str_lit("only int[N] arrays are supported"));
        TcTok lit = cur(p);
        expect(p, TC_TK_INT_LIT, str_lit("expected array length"));
        expect(p, TC_TK_RBRACK, str_lit("expected ']'"));
        s->decl_type.kind = TY_ARRAY_I32;
        s->decl_type.array_len = lit.int_value;
        if (accept(p, TC_TK_LBRACK)) {
            TcTok lit2 = cur(p);
            expect(p, TC_TK_INT_LIT, str_lit("expected array length"));
            expect(p, TC_TK_RBRACK, str_lit("expected ']'"));
            s->decl_type.array_len2 = lit2.int_value;
        }
    }
    if (accept(p, TC_TK_ASSIGN)) {
        s->decl_init = parse_expr(p);
    }
    if (require_semi) expect(p, TC_TK_SEMI, str_lit("expected ';'"));
    return s;
}

static Stmt *parse_stmt(P *p) {
    TcTok t = cur(p);
    if (t.kind == TC_TK_LBRACE) {
        Stmt *s = new_stmt(p, ST_BLOCK, t.line);
        parse_block(p, &s->block_body);
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
    if (t.kind == TC_TK_KW_FOR) {
        p->i++;
        Stmt *s = new_stmt(p, ST_FOR, t.line);
        expect(p, TC_TK_LPAREN, str_lit("expected '('"));
        // init: empty | decl | expr
        if (cur(p).kind == TC_TK_SEMI) {
            p->i++;
            s->for_init = NULL;
        } else if (cur(p).kind == TC_TK_KW_INT || cur(p).kind == TC_TK_KW_FLOAT ||
                   cur(p).kind == TC_TK_KW_CHAR || cur(p).kind == TC_TK_KW_ENUM ||
                   cur(p).kind == TC_TK_KW_CONST || cur(p).kind == TC_TK_KW_VOID ||
                   cur(p).kind == TC_TK_KW_STATIC || cur(p).kind == TC_TK_KW_INLINE ||
                   cur(p).kind == TC_TK_KW_LONG || cur(p).kind == TC_TK_KW_SIGNED ||
                   cur(p).kind == TC_TK_KW_UNSIGNED ||
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
        // step: empty allowed
        if (cur(p).kind != TC_TK_RPAREN) s->for_step = parse_expr(p);
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
    if (t.kind == TC_TK_KW_INT || t.kind == TC_TK_KW_FLOAT ||
        t.kind == TC_TK_KW_STRUCT || t.kind == TC_TK_KW_CHAR ||
        t.kind == TC_TK_KW_ENUM || t.kind == TC_TK_KW_CONST ||
        t.kind == TC_TK_KW_VOID ||
        t.kind == TC_TK_KW_STATIC || t.kind == TC_TK_KW_INLINE ||
        t.kind == TC_TK_KW_LONG || t.kind == TC_TK_KW_SIGNED ||
        t.kind == TC_TK_KW_UNSIGNED ||
        t.kind == TC_TK_KW_VA_LIST ||
        (t.kind == TC_TK_IDENT && typedef_lookup(p, t.text) &&
         peek(p, 1).kind == TC_TK_IDENT)) {
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
    // Accept signed/unsigned/long modifiers in any order ahead of `int` or
    // by themselves. Any 'long' promotes to TY_I64; otherwise TY_I32.
    if (cur(p).kind == TC_TK_KW_SIGNED || cur(p).kind == TC_TK_KW_UNSIGNED ||
        cur(p).kind == TC_TK_KW_LONG) {
        bool saw_long = false;
        while (cur(p).kind == TC_TK_KW_SIGNED || cur(p).kind == TC_TK_KW_UNSIGNED ||
               cur(p).kind == TC_TK_KW_LONG || cur(p).kind == TC_TK_KW_INT) {
            if (cur(p).kind == TC_TK_KW_LONG) saw_long = true;
            p->i++; skip_const(p);
        }
        out->kind = saw_long ? TY_I64 : TY_I32;
        return true;
    }
    if (cur(p).kind == TC_TK_KW_INT)   {
        p->i++; out->kind = TY_I32; skip_const(p);
        if (accept(p, TC_TK_STAR)) {
            out->kind = TY_PTR_I32;
            skip_const(p);
            if (accept(p, TC_TK_STAR)) { wrap_ptr_to_ptr(p, out); skip_const(p); }
        }
        skip_const(p);
        return true;
    }
    if (cur(p).kind == TC_TK_KW_FLOAT) { p->i++; out->kind = TY_F32; skip_const(p); return true; }
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
        }
        skip_const(p);
        return true;
    }
    if (cur(p).kind == TC_TK_KW_STRUCT) {
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
        // `va_list` is only valid as the type of a local variable
        // declaration (handled separately). It is not supported as a
        // parameter / return / cast / sizeof type, because the emitter
        // does not lower it in those positions. Reject up front so that
        // the front-end fails cleanly instead of silently mis-emitting.
        perror_at(p, cur(p).line,
                  str_lit("va_list is not supported in this position "
                          "(only as a local variable declaration)"));
        return false;
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
                if (out->kind == TY_I32) out->kind = TY_PTR_I32;
                else if (out->kind == TY_STRUCT) out->kind = TY_PTR_STRUCT;
                else if (out->kind == TY_FNPTR) {
                    // `BinOp *` where BinOp is a typedef'd function pointer.
                    // Wrap into TY_PTR_PTR(pointee=TY_FNPTR copy).
                    wrap_ptr_to_ptr(p, out);
                }
                // else: ignore (e.g. typedef'd pointer + extra '*' not supported)
                skip_const(p);
                if (accept(p, TC_TK_STAR) &&
                    (out->kind == TY_PTR_I32 || out->kind == TY_PTR_STRUCT)) {
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
    return true;
}

static Func *parse_func(P *p) {
    int line = cur(p).line;
    Type ret_ty = {0};
    if (!parse_sig_type(p, &ret_ty)) {
        perror_at(p, cur(p).line, str_lit("expected return type"));
    }
    TcTok name = cur(p);
    expect(p, TC_TK_IDENT, str_lit("expected function name"));
    expect(p, TC_TK_LPAREN, str_lit("expected '('"));
    Func *f = arena_new(p->arena, Func);
    *f = (Func){0};
    f->name = name.text;
    f->return_type = ret_ty;
    f->line = line;
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
            }
            VecParam_push_back(p->arena, &f->params,
                ((Param){.name = pname, .type = pty, .line = cur(p).line}));
            if (!accept(p, TC_TK_COMMA)) break;
        }
    }
    expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
    if (accept(p, TC_TK_SEMI)) {
        // Forward declaration / prototype: no body.
        f->is_forward = true;
        return f;
    }
    parse_block(p, &f->body);
    return f;
}

static StructDef *parse_struct_def(P *p) {
    int line = cur(p).line;
    expect(p, TC_TK_KW_STRUCT, str_lit("expected 'struct'"));
    TcTok name = cur(p);
    expect(p, TC_TK_IDENT, str_lit("expected struct name"));
    expect(p, TC_TK_LBRACE, str_lit("expected '{'"));
    StructDef *sd = arena_new(p->arena, StructDef);
    *sd = (StructDef){0};
    sd->name = name.text;
    sd->line = line;
    VecStructField_reserve(p->arena, &sd->fields, 4);
    while (cur(p).kind != TC_TK_RBRACE && cur(p).kind != TC_TK_EOF) {
        skip_const(p);
        Type ft = {0};
        bool is_struct_kind = false;
        bool is_ptr = false;
        bool was_char = false;
        bool was_float = false;
        if (cur(p).kind == TC_TK_KW_STRUCT) {
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
        } else {
            TypeKind k;
            was_char = (cur(p).kind == TC_TK_KW_CHAR);
            was_float = (cur(p).kind == TC_TK_KW_FLOAT);
            if (!parse_base_type(p, &k)) {
                perror_at(p, cur(p).line, str_lit("expected field type (int|float|char|struct)"));
                p->i++;
                continue;
            }
            ft.kind = k;
            // Optional pointer suffix on base types.
            if (accept(p, TC_TK_STAR)) {
                is_ptr = true;
                if (was_char) ft.kind = TY_PTR_CHAR;
                else if (k == TY_I32) ft.kind = TY_PTR_I32;
                else if (k == TY_VOID) ft.kind = TY_PTR_VOID;
                else {
                    perror_at(p, cur(p).line, str_lit("only int*/char*/void* pointer fields are supported"));
                }
            } else if (k == TY_VOID) {
                perror_at(p, cur(p).line, str_lit("'void' is not a valid struct field type"));
            }
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
                ft.kind = TY_ARRAY_I32;
                ft.array_len = n1;
                ft.array_len2 = n2;
            }
        }
        expect(p, TC_TK_SEMI, str_lit("expected ';' after field"));
        VecStructField_push_back(p->arena, &sd->fields,
            ((StructField){.name = fn.text, .type = ft}));
    }
    expect(p, TC_TK_RBRACE, str_lit("expected '}'"));
    expect(p, TC_TK_SEMI, str_lit("expected ';' after struct definition"));
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
static void parse_enum_decl_top(P *p) {
    int line = cur(p).line;
    expect(p, TC_TK_KW_ENUM, str_lit("expected 'enum'"));
    // Optional tag — accepted for source compatibility, not stored. The
    // tag namespace is implicit: `enum Tag` always lowers to `int`, so
    // there's nothing to look up later.
    if (cur(p).kind == TC_TK_IDENT) p->i++;
    expect(p, TC_TK_LBRACE, str_lit("expected '{' to begin enum body"));
    int64_t next_value = 0;
    while (cur(p).kind != TC_TK_RBRACE && cur(p).kind != TC_TK_EOF) {
        TcTok nm = cur(p);
        if (!expect(p, TC_TK_IDENT, str_lit("expected enumerator name"))) break;
        int64_t value = next_value;
        if (accept(p, TC_TK_ASSIGN)) {
            bool neg = accept(p, TC_TK_MINUS);
            TcTok lit = cur(p);
            if (lit.kind == TC_TK_INT_LIT) {
                value = neg ? -lit.int_value : lit.int_value;
                p->i++;
            } else if (lit.kind == TC_TK_IDENT && enum_lookup(p, lit.text)) {
                int64_t v = enum_lookup(p, lit.text)->value;
                value = neg ? -v : v;
                p->i++;
            } else {
                perror_at(p, lit.line,
                    str_lit("enum initializer must be an int literal "
                            "or a previously-declared enumerator"));
                p->i++;
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
    expect(p, TC_TK_SEMI, str_lit("expected ';' after enum declaration"));
    (void)line;
}

Program *tinyc_parse(Arena *arena, VecTcTok toks) {
    Program *prog = arena_new(arena, Program);
    *prog = (Program){0};
    tinyc_parse_into(arena, prog, toks);
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

void tinyc_parse_into(Arena *arena, Program *prog, VecTcTok toks) {
    P p = {.arena = arena, .toks = toks.data, .n = toks.size, .i = 0,
           .typedefs = NULL, .enums = NULL};
    // Built-in typedefs (treated as TY_I64, signedness ignored).
    {
        const char *names[] = {"int64_t", "uint64_t", "size_t", "ssize_t",
                               "intptr_t", "uintptr_t", "ptrdiff_t"};
        for (size_t k = 0; k < sizeof(names) / sizeof(names[0]); k++) {
            Typedef *td = arena_new(arena, Typedef);
            td->name = (string){.str = (char *)names[k], .size = 0};
            // Compute string length without libc.
            const char *s = names[k]; size_t len = 0; while (s[len]) len++;
            td->name.size = len;
            td->ty.kind = TY_I64;
            td->next = p.typedefs;
            p.typedefs = td;
        }
        // int32_t / uint32_t — match TY_I32 to keep existing semantics.
        const char *n32[] = {"int32_t", "uint32_t"};
        for (size_t k = 0; k < sizeof(n32) / sizeof(n32[0]); k++) {
            Typedef *td = arena_new(arena, Typedef);
            const char *s = n32[k]; size_t len = 0; while (s[len]) len++;
            td->name = (string){.str = (char *)n32[k], .size = len};
            td->ty.kind = TY_I32;
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
        // Top-level storage-class qualifiers are parser-noise.
        skip_storage_class(&p);
        if (cur(&p).kind == TC_TK_EOF) break;
        if (cur(&p).kind == TC_TK_KW_STRUCT && peek(&p, 2).kind == TC_TK_LBRACE) {
            StructDef *sd = parse_struct_def(&p);
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
            } else if (!struct_def_equal(existing_sd, sd)) {
                perror_at(&p, sd->line,
                    str_lit("conflicting redefinition of struct"));
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
            int line = cur(&p).line;
            p.i++;
            Type ty = {0};
            if (!parse_sig_type(&p, &ty)) {
                perror_at(&p, line, str_lit("expected type after 'typedef'"));
                while (cur(&p).kind != TC_TK_SEMI && cur(&p).kind != TC_TK_EOF) p.i++;
                accept(&p, TC_TK_SEMI);
                continue;
            }
            // Function-pointer typedef: `typedef int (*BinOp)(int, int);`.
            string fnp_name = (string){0};
            if (try_parse_fnptr_suffix(&p, &ty, &fnp_name)) {
                if (fnp_name.size == 0) {
                    perror_at(&p, line, str_lit("function-pointer typedef requires a name"));
                }
                expect(&p, TC_TK_SEMI, str_lit("expected ';' after typedef"));
                Typedef *td = arena_new(arena, Typedef);
                td->name = fnp_name;
                td->ty = ty;
                td->next = p.typedefs;
                p.typedefs = td;
                continue;
            }
            TcTok nm = cur(&p);
            if (!expect(&p, TC_TK_IDENT, str_lit("expected typedef name"))) continue;
            // Optional `[N]` to support `typedef int Vec[N];` (1D only).
            if (accept(&p, TC_TK_LBRACK)) {
                TcTok lit = cur(&p);
                expect(&p, TC_TK_INT_LIT, str_lit("expected array length"));
                expect(&p, TC_TK_RBRACK, str_lit("expected ']'"));
                if (ty.kind == TY_I32) {
                    ty.kind = TY_ARRAY_I32;
                    ty.array_len = lit.int_value;
                } else {
                    perror_at(&p, line, str_lit("typedef array only supported for int"));
                }
            }
            expect(&p, TC_TK_SEMI, str_lit("expected ';' after typedef"));
            Typedef *td = arena_new(arena, Typedef);
            td->name = nm.text;
            td->ty = ty;
            td->next = p.typedefs;
            p.typedefs = td;
            continue;
        }
        // `extern` on a global makes it externally linked (e.g. libc's
        // `stdout`). For function decls it's still parser-noise.
        bool saw_extern = false;
        if (cur(&p).kind == TC_TK_KW_EXTERN) { saw_extern = true; p.i++; }
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
                g.line = cur(&p).line;
                merge_push_global(&p, prog, g);
                continue;
            }
            if (cur(&p).kind == TC_TK_IDENT) {
                TcTokKind after = peek(&p, 1).kind;
                if (after == TC_TK_ASSIGN || after == TC_TK_SEMI) {
                    TcTok nm = cur(&p);
                    p.i++;
                    Global g = (Global){0};
                    g.name = nm.text;
                    g.is_extern = saw_extern;
                    g.type = tty;
                    g.line = nm.line;
                    if (accept(&p, TC_TK_ASSIGN)) {
                        g.has_init = true;
                        TcTok lit = cur(&p);
                        if (lit.kind == TC_TK_INT_LIT) {
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
}

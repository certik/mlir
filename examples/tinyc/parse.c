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

typedef struct {
    Arena *arena;
    TcTok *toks;
    size_t n;
    size_t i;
    Typedef *typedefs;
} P;

static TcTok cur(P *p) { return p->toks[p->i]; }
static TcTok peek(P *p, size_t k) { return p->toks[p->i + k < p->n ? p->i + k : p->n - 1]; }

static Typedef *typedef_lookup(P *p, string name) {
    for (Typedef *t = p->typedefs; t; t = t->next) {
        if (str_eq(t->name, name)) return t;
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
        Type ty = {0};
        if (!parse_sig_type(p, &ty)) {
            perror_at(p, cur(p).line, str_lit("expected type in sizeof(...)"));
        }
        expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
        Expr *e = new_expr(p, EX_SIZEOF, t.line);
        e->cast_type = ty;
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
                        nxt == TC_TK_KW_STRUCT || nxt == TC_TK_KW_CHAR);
        if (!is_cast && nxt == TC_TK_IDENT &&
            typedef_lookup(p, peek(p, 1).text)) {
            is_cast = true;
        }
        if (is_cast) {
            p->i++;  // consume '('
            Type ty = {0};
            if (!parse_sig_type(p, &ty)) {
                perror_at(p, cur(p).line, str_lit("expected type in cast"));
            }
            expect(p, TC_TK_RPAREN, str_lit("expected ')' in cast"));
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
    if (cur(p).kind == TC_TK_KW_INT)   { p->i++; *out = TY_I32; return true; }
    if (cur(p).kind == TC_TK_KW_FLOAT) { p->i++; *out = TY_F32; return true; }
    if (cur(p).kind == TC_TK_KW_CHAR)  { p->i++; *out = TY_I32; return true; }
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
        } else {
            s->decl_type.kind = is_ptr ? TY_PTR_STRUCT : TY_STRUCT;
        }
        if (accept(p, TC_TK_ASSIGN)) {
            if (s->decl_type.kind != TY_PTR_STRUCT) {
                perror_at(p, line, str_lit("struct/array initializers are not supported"));
            }
            s->decl_init = parse_expr(p);
        }
        if (require_semi) expect(p, TC_TK_SEMI, str_lit("expected ';'"));
        return s;
    }
    TypeKind base = TY_I32;
    bool was_char = (cur(p).kind == TC_TK_KW_CHAR);
    parse_base_type(p, &base);
    bool is_ptr = false;
    if (accept(p, TC_TK_STAR)) is_ptr = true;
    TcTok name = cur(p);
    expect(p, TC_TK_IDENT, str_lit("expected identifier"));
    Stmt *s = new_stmt(p, ST_DECL, line);
    s->decl_name = name.text;
    s->decl_type.kind = base;
    if (is_ptr) {
        if (was_char) s->decl_type.kind = TY_PTR_CHAR;
        else if (base == TY_I32) s->decl_type.kind = TY_PTR_I32;
        else perror_at(p, line, str_lit("only int*/char* pointers are supported"));
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
        s->expr = parse_expr(p);
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
                   cur(p).kind == TC_TK_KW_CHAR ||
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
                expect(p, TC_TK_INT_LIT, str_lit("expected integer in 'case'"));
                expect(p, TC_TK_COLON, str_lit("expected ':' after case"));
                SwitchCase sc = (SwitchCase){
                    .value = neg ? -lit.int_value : lit.int_value,
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
        (t.kind == TC_TK_IDENT && typedef_lookup(p, t.text) &&
         peek(p, 1).kind == TC_TK_IDENT)) {
        return parse_decl(p, /*require_semi*/ true);
    }
    // expression statement (covers assignments and calls)
    Stmt *s = new_stmt(p, ST_EXPR, t.line);
    s->expr = parse_expr(p);
    expect(p, TC_TK_SEMI, str_lit("expected ';'"));
    return s;
}

// Parse a function-signature type:  int | float | char [*] | struct Name [*]
// or a typedef'd type-name (in which case any leading '*' for typedef-int
// produces TY_PTR_I32, etc.).
// Returns false if the current tokens don't start a type.
static bool parse_sig_type(P *p, Type *out) {
    *out = (Type){0};
    if (cur(p).kind == TC_TK_KW_INT)   { p->i++; out->kind = TY_I32; if (accept(p, TC_TK_STAR)) out->kind = TY_PTR_I32; return true; }
    if (cur(p).kind == TC_TK_KW_FLOAT) { p->i++; out->kind = TY_F32; return true; }
    if (cur(p).kind == TC_TK_KW_CHAR)  {
        p->i++;
        if (accept(p, TC_TK_STAR)) out->kind = TY_PTR_CHAR;
        else out->kind = TY_I32;
        return true;
    }
    if (cur(p).kind == TC_TK_KW_STRUCT) {
        p->i++;
        TcTok sn = cur(p);
        if (!expect(p, TC_TK_IDENT, str_lit("expected struct name"))) return false;
        out->kind = TY_STRUCT;
        out->struct_name = sn.text;
        if (accept(p, TC_TK_STAR)) out->kind = TY_PTR_STRUCT;
        return true;
    }
    if (cur(p).kind == TC_TK_IDENT) {
        Typedef *td = typedef_lookup(p, cur(p).text);
        if (td) {
            p->i++;
            *out = td->ty;
            // Allow `Td*` to add another level of pointer (only for scalar
            // typedefs; struct typedefs already encode the pointer).
            if (accept(p, TC_TK_STAR)) {
                if (out->kind == TY_I32) out->kind = TY_PTR_I32;
                else if (out->kind == TY_STRUCT) out->kind = TY_PTR_STRUCT;
                // else: ignore (e.g. typedef'd pointer + extra '*' not supported)
            }
            return true;
        }
    }
    return false;
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
    if (cur(p).kind != TC_TK_RPAREN) {
        for (;;) {
            Type pty = {0};
            if (!parse_sig_type(p, &pty)) {
                perror_at(p, cur(p).line, str_lit("expected parameter type"));
            }
            string pname = (string){0};
            if (cur(p).kind == TC_TK_IDENT) {
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
        Type ft = {0};
        if (cur(p).kind == TC_TK_KW_STRUCT) {
            // Nested struct field: `struct Inner name;` (by-value) or
            // `struct Inner* name;` (pointer to struct).
            p->i++;
            TcTok sn = cur(p);
            if (!expect(p, TC_TK_IDENT, str_lit("expected struct name"))) { p->i++; continue; }
            ft.kind = accept(p, TC_TK_STAR) ? TY_PTR_STRUCT : TY_STRUCT;
            ft.struct_name = sn.text;
        } else {
            TypeKind k;
            if (!parse_base_type(p, &k)) {
                perror_at(p, cur(p).line, str_lit("expected field type (int|float|struct)"));
                p->i++;
                continue;
            }
            ft.kind = k;
        }
        TcTok fn = cur(p);
        expect(p, TC_TK_IDENT, str_lit("expected field name"));
        expect(p, TC_TK_SEMI, str_lit("expected ';' after field"));
        VecStructField_push_back(p->arena, &sd->fields,
            ((StructField){.name = fn.text, .type = ft}));
    }
    expect(p, TC_TK_RBRACE, str_lit("expected '}'"));
    expect(p, TC_TK_SEMI, str_lit("expected ';' after struct definition"));
    return sd;
}

Program *tinyc_parse(Arena *arena, VecTcTok toks) {
    P p = {.arena = arena, .toks = toks.data, .n = toks.size, .i = 0, .typedefs = NULL};
    Program *prog = arena_new(arena, Program);
    *prog = (Program){0};
    VecFuncPtr_reserve(arena, &prog->funcs, 4);
    VecStructDefPtr_reserve(arena, &prog->structs, 4);
    VecGlobal_reserve(arena, &prog->globals, 4);
    while (cur(&p).kind != TC_TK_EOF) {
        if (cur(&p).kind == TC_TK_KW_STRUCT && peek(&p, 2).kind == TC_TK_LBRACE) {
            StructDef *sd = parse_struct_def(&p);
            VecStructDefPtr_push_back(arena, &prog->structs, sd);
            continue;
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
        // `extern` is parser-noise: drop the keyword and parse the rest as
        // a normal forward decl / global.
        if (cur(&p).kind == TC_TK_KW_EXTERN) p.i++;
        // Disambiguate top-level decl between a function and a global.
        // We look ahead past `<type>` (and optional `*`) for an IDENT
        // followed by `=` or `;` -> global, otherwise function.
        size_t save = p.i;
        Type tty = {0};
        if (parse_sig_type(&p, &tty)) {
            if (cur(&p).kind == TC_TK_IDENT) {
                TcTokKind after = peek(&p, 1).kind;
                if (after == TC_TK_ASSIGN || after == TC_TK_SEMI) {
                    TcTok nm = cur(&p);
                    p.i++;
                    Global g = (Global){0};
                    g.name = nm.text;
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
                        } else {
                            perror_at(&p, lit.line,
                                str_lit("global initializer must be a literal"));
                            p.i++;
                        }
                    }
                    expect(&p, TC_TK_SEMI, str_lit("expected ';' after global"));
                    VecGlobal_push_back(arena, &prog->globals, g);
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
        } else if (!existing->is_forward && f->is_forward) {
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
    return prog;
}

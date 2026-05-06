// tinyC parser — recursive descent. Produces an AST defined in tinyc.h.

#include "tinyc.h"

#include <base/arena.h>
#include <base/io.h>
#include <base/string.h>

typedef struct {
    Arena *arena;
    TcTok *toks;
    size_t n;
    size_t i;
} P;

static TcTok cur(P *p) { return p->toks[p->i]; }
static TcTok peek(P *p, size_t k) { return p->toks[p->i + k < p->n ? p->i + k : p->n - 1]; }

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

// postfix := (...) | [expr]
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
    if (t.kind == TC_TK_LPAREN) {
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
static Expr *parse_rel(P *p) {
    Expr *l = parse_add(p);
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
        Expr *r = parse_add(p);
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
static Expr *parse_and(P *p) {
    Expr *l = parse_eq(p);
    while (cur(p).kind == TC_TK_AMPAMP) {
        int line = cur(p).line;
        p->i++;
        Expr *r = parse_eq(p);
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
static Expr *parse_assign_or_or(P *p) {
    // Parse a logical-or expression first, then if '=' follows, treat
    // the LHS as an lvalue (validated in the emitter).
    Expr *lhs = parse_or(p);
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

// Parse a decl statement: <type> [*] name [ '[' N ']' ] [ '=' expr ] ';'
// Caller has already verified that current token starts a base type.
static Stmt *parse_decl(P *p, bool require_semi) {
    int line = cur(p).line;
    TypeKind base = TY_I32;
    parse_base_type(p, &base);
    bool is_ptr = false;
    if (accept(p, TC_TK_STAR)) is_ptr = true;
    TcTok name = cur(p);
    expect(p, TC_TK_IDENT, str_lit("expected identifier"));
    Stmt *s = new_stmt(p, ST_DECL, line);
    s->decl_name = name.text;
    s->decl_type.kind = base;
    if (is_ptr) {
        if (base != TY_I32) perror_at(p, line, str_lit("only int* pointers are supported"));
        s->decl_type.kind = TY_PTR_I32;
    }
    if (accept(p, TC_TK_LBRACK)) {
        if (base != TY_I32 || is_ptr) perror_at(p, line, str_lit("only int[N] arrays are supported"));
        TcTok lit = cur(p);
        expect(p, TC_TK_INT_LIT, str_lit("expected array length"));
        expect(p, TC_TK_RBRACK, str_lit("expected ']'"));
        s->decl_type.kind = TY_ARRAY_I32;
        s->decl_type.array_len = lit.int_value;
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
        } else if (cur(p).kind == TC_TK_KW_INT || cur(p).kind == TC_TK_KW_FLOAT) {
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
    if (t.kind == TC_TK_KW_INT || t.kind == TC_TK_KW_FLOAT) {
        return parse_decl(p, /*require_semi*/ true);
    }
    // expression statement (covers assignments and calls)
    Stmt *s = new_stmt(p, ST_EXPR, t.line);
    s->expr = parse_expr(p);
    expect(p, TC_TK_SEMI, str_lit("expected ';'"));
    return s;
}

static Func *parse_func(P *p) {
    int line = cur(p).line;
    expect(p, TC_TK_KW_INT, str_lit("expected return type 'int'"));
    TcTok name = cur(p);
    expect(p, TC_TK_IDENT, str_lit("expected function name"));
    expect(p, TC_TK_LPAREN, str_lit("expected '('"));
    Func *f = arena_new(p->arena, Func);
    *f = (Func){0};
    f->name = name.text;
    f->line = line;
    VecParam_reserve(p->arena, &f->params, 4);
    VecStmtPtr_reserve(p->arena, &f->body, 8);
    if (cur(p).kind != TC_TK_RPAREN) {
        for (;;) {
            expect(p, TC_TK_KW_INT, str_lit("expected 'int'"));
            TcTok pn = cur(p);
            expect(p, TC_TK_IDENT, str_lit("expected parameter name"));
            VecParam_push_back(p->arena, &f->params, ((Param){.name = pn.text, .line = pn.line}));
            if (!accept(p, TC_TK_COMMA)) break;
        }
    }
    expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
    parse_block(p, &f->body);
    return f;
}

Program *tinyc_parse(Arena *arena, VecTcTok toks) {
    P p = {.arena = arena, .toks = toks.data, .n = toks.size, .i = 0};
    Program *prog = arena_new(arena, Program);
    *prog = (Program){0};
    VecFuncPtr_reserve(arena, &prog->funcs, 4);
    while (cur(&p).kind != TC_TK_EOF) {
        Func *f = parse_func(&p);
        VecFuncPtr_push_back(arena, &prog->funcs, f);
    }
    return prog;
}

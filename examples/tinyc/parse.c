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
    VecStmtPtr_reserve(p->arena, &s->block_body, 4);
    return s;
}

static Expr *parse_expr(P *p);

// primary := INT_LIT | IDENT | IDENT '(' args ')' | '(' expr ')' | unary
static Expr *parse_primary(P *p) {
    TcTok t = cur(p);
    if (t.kind == TC_TK_INT_LIT) {
        p->i++;
        Expr *e = new_expr(p, EX_INT, t.line);
        e->int_value = t.int_value;
        return e;
    }
    if (t.kind == TC_TK_LPAREN) {
        p->i++;
        Expr *e = parse_expr(p);
        expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
        return e;
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
            return e;
        }
        Expr *e = new_expr(p, EX_VAR, t.line);
        e->name = t.text;
        return e;
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
    // Handle `IDENT '=' assign` if pattern matches; else fall through to OR.
    if (cur(p).kind == TC_TK_IDENT && peek(p, 1).kind == TC_TK_ASSIGN) {
        TcTok name = cur(p);
        p->i += 2;
        Expr *rhs = parse_assign_or_or(p);   // right-assoc
        Expr *e = new_expr(p, EX_ASSIGN, name.line);
        e->name = name.text;
        e->rhs_assign = rhs;
        return e;
    }
    return parse_or(p);
}

static Expr *parse_expr(P *p) { return parse_assign_or_or(p); }

static Stmt *parse_stmt(P *p);

static void parse_block(P *p, VecStmtPtr *out) {
    expect(p, TC_TK_LBRACE, str_lit("expected '{'"));
    while (cur(p).kind != TC_TK_RBRACE && cur(p).kind != TC_TK_EOF) {
        Stmt *s = parse_stmt(p);
        VecStmtPtr_push_back(p->arena, out, s);
    }
    expect(p, TC_TK_RBRACE, str_lit("expected '}'"));
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
    if (t.kind == TC_TK_KW_PRINT) {
        p->i++;
        Stmt *s = new_stmt(p, ST_PRINT, t.line);
        expect(p, TC_TK_LPAREN, str_lit("expected '('"));
        s->expr = parse_expr(p);
        expect(p, TC_TK_RPAREN, str_lit("expected ')'"));
        expect(p, TC_TK_SEMI, str_lit("expected ';'"));
        return s;
    }
    if (t.kind == TC_TK_KW_INT) {
        p->i++;
        TcTok name = cur(p);
        expect(p, TC_TK_IDENT, str_lit("expected identifier"));
        Stmt *s = new_stmt(p, ST_DECL, t.line);
        s->decl_name = name.text;
        if (accept(p, TC_TK_ASSIGN)) {
            s->decl_init = parse_expr(p);
        }
        expect(p, TC_TK_SEMI, str_lit("expected ';'"));
        return s;
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

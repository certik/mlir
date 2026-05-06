// tinyC — a tiny subset of C compiled to MLIR via mlir_api.h only.
//
// This file is intentionally self-contained and uses ONLY mlir_api.h
// plus corec primitives. It must NEVER include anything from
// upstream MLIR directly; the whole point of this example is to prove
// that a compiler can be written against the public API.
//
// Subset of C supported in this iteration:
//   - Types: int (i32) only
//   - Local variables (declared with `int`, mutable)
//   - Integer literals
//   - Binary ops: + - * / %
//   - Comparisons: < <= > >= == !=
//   - Logical: && || ! (lowered via short-circuit + select)
//   - Assignments
//   - if / else, while
//   - Functions with int parameters returning int
//   - `print(expr);` builtin -> vector.print
//   - Top-level entry point: int main()
//
// Not yet supported: arrays, pointers, structs, char/string, float,
// for-loops, break/continue.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <base/arena.h>
#include <base/string.h>

#include "mlir_api.h"

// ---------------- AST ----------------

typedef enum {
    EX_INT,            // integer literal
    EX_VAR,            // variable reference
    EX_BIN,            // binary op (kind in `op`)
    EX_UN,             // unary op
    EX_CALL,           // function call
    EX_ASSIGN,         // var = expr
} ExprKind;

typedef enum {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
    OP_LT, OP_LE, OP_GT, OP_GE, OP_EQ, OP_NE,
    OP_AND, OP_OR,                // logical && || (short-circuit)
    OP_NEG, OP_NOT,               // unary - !
} BinOp;

typedef struct Expr Expr;

DEFINE_VECTOR_FOR_TYPE(Expr*, VecExprPtr)

struct Expr {
    ExprKind kind;
    // For EX_INT
    int64_t int_value;
    // For EX_VAR / EX_ASSIGN
    string name;
    // For EX_BIN / EX_UN
    BinOp op;
    Expr *lhs;
    Expr *rhs;
    // For EX_CALL
    string callee;
    VecExprPtr args;
    // For EX_ASSIGN
    Expr *rhs_assign;
    // Source line for diagnostics
    int line;
};

typedef enum {
    ST_EXPR,           // expression-statement (e.g. f(x);)
    ST_DECL,           // int x = expr;
    ST_RETURN,         // return expr;
    ST_IF,             // if (c) { then } [else { else }]
    ST_WHILE,          // while (c) { body }
    ST_BLOCK,          // { ... }
    ST_PRINT,          // print(expr);
} StmtKind;

typedef struct Stmt Stmt;

DEFINE_VECTOR_FOR_TYPE(Stmt*, VecStmtPtr)

struct Stmt {
    StmtKind kind;
    // ST_EXPR / ST_RETURN / ST_PRINT
    Expr *expr;
    // ST_DECL
    string decl_name;
    Expr *decl_init;
    // ST_IF
    Expr *cond;
    VecStmtPtr then_body;
    VecStmtPtr else_body;
    bool has_else;
    // ST_WHILE
    VecStmtPtr while_body;
    // ST_BLOCK
    VecStmtPtr block_body;
    int line;
};

typedef struct {
    string name;
    int line;
} Param;

DEFINE_VECTOR_FOR_TYPE(Param, VecParam)

typedef struct {
    string name;
    VecParam params;
    VecStmtPtr body;
    int line;
} Func;

DEFINE_VECTOR_FOR_TYPE(Func*, VecFuncPtr)

typedef struct {
    VecFuncPtr funcs;
} Program;

// ---------------- Lexer ----------------

typedef enum {
    TC_TK_EOF = 0,
    TC_TK_INT_LIT,
    TC_TK_IDENT,
    TC_TK_KW_INT,
    TC_TK_KW_RETURN,
    TC_TK_KW_IF,
    TC_TK_KW_ELSE,
    TC_TK_KW_WHILE,
    TC_TK_KW_PRINT,
    TC_TK_LPAREN, TC_TK_RPAREN,
    TC_TK_LBRACE, TC_TK_RBRACE,
    TC_TK_SEMI, TC_TK_COMMA,
    TC_TK_PLUS, TC_TK_MINUS, TC_TK_STAR, TC_TK_SLASH, TC_TK_PERCENT,
    TC_TK_LT, TC_TK_LE, TC_TK_GT, TC_TK_GE, TC_TK_EQEQ, TC_TK_NE,
    TC_TK_ASSIGN, TC_TK_BANG,
    TC_TK_AMPAMP, TC_TK_PIPEPIPE,
} TcTokKind;

typedef struct {
    TcTokKind kind;
    int64_t int_value;
    string text;             // interned identifier text (for IDENT)
    int line;
} TcTok;

DEFINE_VECTOR_FOR_TYPE(TcTok, VecTcTok)

VecTcTok tinyc_lex(Arena *arena, string src);

// ---------------- Parser ----------------

Program *tinyc_parse(Arena *arena, VecTcTok toks);

// ---------------- Emitter ----------------

// Build a top-level MLIR module from `program`. The returned op handle
// is the `builtin.module` op. Uses only mlir_api.h.
MLIR_OpHandle tinyc_emit_module(MLIR_Context *ctx, Program *program);

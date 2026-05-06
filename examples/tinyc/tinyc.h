// tinyC — a tiny subset of C compiled to MLIR via mlir_api.h only.
//
// This file is intentionally self-contained and uses ONLY mlir_api.h
// plus corec primitives. It must NEVER include anything from
// upstream MLIR directly; the whole point of this example is to prove
// that a compiler can be written against the public API.
//
// Subset of C supported:
//   - Types: int (i32), float (f32), int[N], int* (alias-only),
//     struct S { int / float / nested-struct fields },
//     struct S* (alias-only), struct S[N] (fixed-size array of struct)
//   - Local variables (mutable)
//   - Integer and float literals
//   - Binary ops: + - * / (% on int only)
//   - Comparisons: < <= > >= == != (int and float)
//   - Logical: && || ! (short-circuit, int operands)
//   - Assignments to general lvalues: x, a[i], *p, s.field, p->field,
//     s.inner.x (nested), arr[i].x (array of struct)
//   - if / else, while, for
//   - break, continue, early return
//   - Address-of (&x, &s) and dereference (*p) — alias-only pointers
//   - Functions with int / float / struct / struct* parameters and
//     int / float / struct return types.
//     Struct params and returns are scalarized at the function boundary
//     (one MLIR scalar per LEAF field, in declaration order — Clang-style
//     ABI lowering, recursive through nested structs).
//   - `print(expr);` builtin -> vector.print
//   - Top-level entry point: int main()
//
// Pointer-mediated recursive data structures:
//   - struct fields can be `struct Foo*` (TY_PTR_STRUCT). By-value cycles
//     in struct definitions are detected and rejected at emit time.
//   - `null` literal: a polymorphic null pointer (lowered to
//     llvm.mlir.zero : !llvm.ptr).
//   - `sizeof(<type>)` operator: compile-time i32 with a fixed layout
//     (i32/f32 = 4, ptr = 8, struct = padded sum of fields).
//   - C-style cast `(T*)expr` between pointer types: a no-op (we use
//     opaque `!llvm.ptr`).
//   - Built-in extern functions `malloc(i64) -> !llvm.ptr` and
//     `free(!llvm.ptr)` are always declared at module scope and linked
//     against libc.
//   - Pointer comparisons `p == null`, `p != null`, `p == q` use
//     llvm.icmp.
//
// Not supported: strings, int* reassignment, pointer arithmetic,
// arrays-of-pointer, function pointers, returning struct*, array-of-
// struct as a function parameter or return, &<sub-struct-field>, struct
// literal initialization, struct copy `q = p;` (use a wrapper or
// field-by-field assignment).
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <base/arena.h>
#include <base/string.h>

#include "mlir_api.h"

// ---------------- AST ----------------

// Type kinds known to the parser. The emitter computes per-expression
// MLIR types on the fly; this tag is just enough to (a) decide what
// memref to allocate for a declaration, (b) drive lvalue/address mode,
// and (c) decide between integer/float arithmetic ops in the emitter.
typedef enum {
    TY_I32,
    TY_F32,
    TY_PTR_I32,        // alias-only pointer to int
    TY_ARRAY_I32,      // fixed-size int[N], length in `array_len`
    TY_STRUCT,         // struct value (fields stored as flat per-leaf scalars)
    TY_PTR_STRUCT,     // alias-only pointer to struct (bundle of memref aliases)
    TY_ARRAY_STRUCT,   // fixed-size struct[N], length in `array_len`
} TypeKind;

typedef struct {
    TypeKind kind;
    int64_t  array_len;
    string   struct_name;     // for TY_STRUCT
} Type;

typedef enum {
    EX_INT,            // integer literal
    EX_FLOAT,          // float literal
    EX_NULL,           // null pointer literal
    EX_SIZEOF,         // sizeof(<type>) — uses cast_type
    EX_CAST,           // (T*)expr  — uses cast_type and lhs
    EX_VAR,            // variable reference (lvalue when used as such)
    EX_BIN,            // binary op (kind in `op`)
    EX_UN,             // unary op
    EX_CALL,           // function call
    EX_ASSIGN,         // lvalue = expr
    EX_INDEX,          // a[i]
    EX_ADDR,           // &x
    EX_DEREF,          // *p
    EX_FIELD,          // s.x  (lhs = struct lvalue, name = field name)
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
    // For EX_FLOAT
    double float_value;
    // For EX_VAR
    string name;
    // For EX_BIN / EX_UN
    BinOp op;
    Expr *lhs;
    Expr *rhs;
    // For EX_CALL
    string callee;
    VecExprPtr args;
    // For EX_ASSIGN
    Expr *lvalue;          // an lvalue expression (EX_VAR / EX_INDEX / EX_DEREF)
    Expr *rhs_assign;
    // For EX_INDEX (lhs = array, rhs = index)
    // For EX_ADDR / EX_DEREF (lhs = inner)
    // For EX_SIZEOF / EX_CAST: cast_type is the type being asked about /
    // cast to. EX_CAST also uses lhs as the operand.
    Type cast_type;
    int line;
};

typedef enum {
    ST_EXPR,           // expression-statement (e.g. f(x);)
    ST_DECL,           // <type> name [= expr];
    ST_RETURN,         // return expr;
    ST_IF,             // if (c) { then } [else { else }]
    ST_WHILE,          // while (c) { body }
    ST_FOR,            // for (init; cond; step) body
    ST_BLOCK,          // { ... }
    ST_PRINT,          // print(expr);
    ST_BREAK,          // break;
    ST_CONTINUE,       // continue;
} StmtKind;

typedef struct Stmt Stmt;

DEFINE_VECTOR_FOR_TYPE(Stmt*, VecStmtPtr)

struct Stmt {
    StmtKind kind;
    // ST_EXPR / ST_RETURN / ST_PRINT
    Expr *expr;
    // ST_DECL
    Type  decl_type;
    string decl_name;
    Expr *decl_init;
    // ST_IF / ST_WHILE / ST_FOR
    Expr *cond;
    VecStmtPtr then_body;
    VecStmtPtr else_body;
    bool has_else;
    VecStmtPtr while_body;
    // ST_FOR
    Stmt *for_init;
    Expr *for_step;
    VecStmtPtr for_body;
    // ST_BLOCK
    VecStmtPtr block_body;
    int line;
};

typedef struct {
    string name;
    Type   type;             // parameter type (TY_I32 / TY_F32 / TY_STRUCT)
    int line;
} Param;

DEFINE_VECTOR_FOR_TYPE(Param, VecParam)

typedef struct {
    string     name;
    Type       return_type;  // function return type
    VecParam   params;
    VecStmtPtr body;
    int        line;
} Func;

DEFINE_VECTOR_FOR_TYPE(Func*, VecFuncPtr)

typedef struct {
    string name;
    Type   type;        // field type. May be TY_I32, TY_F32, TY_STRUCT
                        // (nested by-value struct), or TY_PTR_STRUCT
                        // (pointer to struct, possibly self-referencing).
                        // TY_ARRAY_STRUCT is not allowed in fields yet.
} StructField;

DEFINE_VECTOR_FOR_TYPE(StructField, VecStructField)

typedef struct StructDef {
    string         name;
    VecStructField fields;
    int            line;
} StructDef;

DEFINE_VECTOR_FOR_TYPE(StructDef*, VecStructDefPtr)

typedef struct {
    VecFuncPtr      funcs;
    VecStructDefPtr structs;
} Program;

// ---------------- Lexer ----------------

typedef enum {
    TC_TK_EOF = 0,
    TC_TK_INT_LIT,
    TC_TK_FLOAT_LIT,
    TC_TK_IDENT,
    TC_TK_KW_INT,
    TC_TK_KW_FLOAT,
    TC_TK_KW_RETURN,
    TC_TK_KW_IF,
    TC_TK_KW_ELSE,
    TC_TK_KW_WHILE,
    TC_TK_KW_FOR,
    TC_TK_KW_BREAK,
    TC_TK_KW_CONTINUE,
    TC_TK_KW_PRINT,
    TC_TK_KW_STRUCT,
    TC_TK_KW_NULL,
    TC_TK_KW_SIZEOF,
    TC_TK_LPAREN, TC_TK_RPAREN,
    TC_TK_LBRACE, TC_TK_RBRACE,
    TC_TK_LBRACK, TC_TK_RBRACK,
    TC_TK_SEMI, TC_TK_COMMA, TC_TK_DOT,
    TC_TK_PLUS, TC_TK_MINUS, TC_TK_STAR, TC_TK_SLASH, TC_TK_PERCENT,
    TC_TK_LT, TC_TK_LE, TC_TK_GT, TC_TK_GE, TC_TK_EQEQ, TC_TK_NE,
    TC_TK_ASSIGN, TC_TK_BANG,
    TC_TK_AMP,
    TC_TK_AMPAMP, TC_TK_PIPEPIPE,
    TC_TK_ARROW,                  // ->  (sugar for (*p).field)
} TcTokKind;

typedef struct {
    TcTokKind kind;
    int64_t int_value;
    double  float_value;
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

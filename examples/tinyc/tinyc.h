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
//     int / float / struct / struct* return types.
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
// Tier-1 extensions:
//   - String literals "..." (with backslash escapes \n \t \r \\ \" \0)
//     evaluate to a `char*` (lowered as `!llvm.ptr` to a private
//     `!llvm.array<N x i8>` constant emitted at module scope). Pass them
//     to `print(...)` to get string output (newline appended), or to a
//     `char*` parameter / global.
//   - Module-scope globals: `int g; int g = 42; float f = 1.5; char *m = "...";`
//     are emitted as `llvm.mlir.global` and read/written via
//     `llvm.mlir.addressof` + `llvm.load`/`llvm.store`. Initializers must
//     be literals (int / float / `null` / string).
//   - `char` keyword + `char*` type. Standalone `char` arithmetic is NOT
//     supported; the only use of `char` in this PR is via `char*`
//     pointers, populated from string literals / casts, and printed.
//   - Compound assignment: `+= -= *= /= %= &= |= ^= <<= >>=` desugar to
//     `lhs = lhs OP rhs` (the lvalue is evaluated twice — keep it
//     simple: a plain variable / field / `arr[i]` with side-effect-free
//     index).
//   - `++x` / `--x` (prefix) and `x++` / `x--` (postfix) desugar to
//     `x = x ± 1`. Both forms produce the *new* value (post-increment
//     value semantics differ from C; only safe in expression-statements
//     and for-step contexts).
//   - Bitwise `& | ^ ~` and shifts `<< >>` on `int` (i32) with C
//     precedence (`<< >>` between `+ -` and `<`; `&` `^` `|` between
//     `==` and `&&`).
//   - Ternary `cond ? a : b` lowering via `cf.cond_br` + a merge block
//     with one block-arg (same shape as the existing `&&`/`||` lowering).
//
// Not supported: int* reassignment, pointer arithmetic, arrays-of-pointer,
// function pointers, returning struct*, array-of-struct as a function
// parameter or return, &<sub-struct-field>, struct literal initialization,
// struct copy `q = p;` (use a wrapper or field-by-field assignment),
// reading individual chars out of a `char*` (`p[i]`), char[N] arrays,
// expression-level char arithmetic.
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
    TY_PTR_CHAR,       // pointer to char (string literals / globals)
    TY_ARRAY_I32,      // fixed-size int[N] / int[N][M] (length(s) in array_len[2])
    TY_ARRAY_F32,      // fixed-size float[N] / float[N][M] (struct field only)
    TY_ARRAY_PTR_STRUCT, // fixed-size (struct T*)[N] (struct field only;
                       // struct_name carries T)
    TY_ARRAY_PTR_CHAR, // fixed-size (char*)[N] (struct field only)
    TY_STRUCT,         // struct value (fields stored as flat per-leaf scalars)
    TY_PTR_STRUCT,     // alias-only pointer to struct (bundle of memref aliases)
    TY_ARRAY_STRUCT,   // fixed-size struct[N], length in `array_len`
    TY_FNPTR,          // function pointer with a fixed signature
} TypeKind;

typedef struct Type Type;
struct Type {
    TypeKind kind;
    int64_t  array_len;       // 1st dimension for TY_ARRAY_I32 / TY_ARRAY_STRUCT
    int64_t  array_len2;      // 2nd dimension for TY_ARRAY_I32 (0 = 1D)
    string   struct_name;     // for TY_STRUCT
    // For TY_FNPTR: arena-allocated return type and parameter types.
    // Storage type is always !llvm.ptr; this signature is consulted at
    // indirect-call sites to build the func.call_indirect operand types.
    Type    *fnptr_ret;
    Type    *fnptr_params;
    int      fnptr_nparams;
};

typedef enum {
    EX_INT,            // integer literal
    EX_FLOAT,          // float literal
    EX_STR,            // string literal (name = decoded bytes incl. NUL)
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
    EX_TERNARY,        // c ? a : b — lhs=cond, rhs=then, lvalue=else
} ExprKind;

typedef enum {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
    OP_LT, OP_LE, OP_GT, OP_GE, OP_EQ, OP_NE,
    OP_AND, OP_OR,                // logical && || (short-circuit)
    OP_BAND, OP_BOR, OP_BXOR,     // bitwise & | ^
    OP_SHL, OP_SHR,               // shifts << >>
    OP_NEG, OP_NOT,               // unary - !
    OP_BNOT,                      // unary ~
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
    ST_SWITCH,         // switch (e) { case 1: ... default: ... }
} StmtKind;

typedef struct Stmt Stmt;

DEFINE_VECTOR_FOR_TYPE(Stmt*, VecStmtPtr)

// One entry in a `switch` body: marks the start of a case-block. The
// body (Stmt array on the parent ST_SWITCH) is parsed as a flat sequence;
// each case label / default record points at the index in that sequence
// where the case body begins. Multiple case labels with the same body
// (`case 1: case 2:`) produce multiple SwitchCase entries with the same
// body_start.
typedef struct {
    int64_t value;
    bool    is_default;
    size_t  body_start;
} SwitchCase;

DEFINE_VECTOR_FOR_TYPE(SwitchCase, VecSwitchCase)

struct Stmt {
    StmtKind kind;
    // ST_EXPR / ST_RETURN / ST_PRINT
    Expr *expr;
    // ST_DECL
    Type  decl_type;
    string decl_name;
    Expr *decl_init;
    // ST_IF / ST_WHILE / ST_FOR / ST_SWITCH
    Expr *cond;
    VecStmtPtr then_body;
    VecStmtPtr else_body;
    bool has_else;
    VecStmtPtr while_body;
    // ST_FOR
    Stmt *for_init;
    Expr *for_step;
    VecStmtPtr for_body;
    // ST_BLOCK / ST_SWITCH (body)
    VecStmtPtr block_body;
    // ST_SWITCH
    VecSwitchCase switch_cases;
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
    bool       is_forward;   // true when this Func is just a prototype
                             // (no body). Replaced in-place by a later
                             // definition with the same name.
    int        line;
} Func;

DEFINE_VECTOR_FOR_TYPE(Func*, VecFuncPtr)

typedef struct {
    string name;
    Type   type;        // field type. May be TY_I32, TY_F32, TY_STRUCT
                        // (nested by-value struct), TY_PTR_STRUCT
                        // (pointer to struct, possibly self-referencing),
                        // TY_PTR_CHAR (char*), or one of the array kinds:
                        // TY_ARRAY_I32 (1D / 2D int), TY_ARRAY_F32 (1D / 2D
                        // float), TY_ARRAY_PTR_STRUCT (struct T*[N]),
                        // TY_ARRAY_PTR_CHAR (char*[N]).
                        // TY_ARRAY_STRUCT is not allowed in fields yet.
} StructField;

DEFINE_VECTOR_FOR_TYPE(StructField, VecStructField)

typedef struct StructDef {
    string         name;
    VecStructField fields;
    int            line;
} StructDef;

DEFINE_VECTOR_FOR_TYPE(StructDef*, VecStructDefPtr)

// Module-scope global variable. Restricted to literal initializers:
//   int g;          (zero)
//   int g = 42;     (init_int)
//   float f = 1.5;  (init_float)
//   char *s = "hi"; (init_str carries the decoded bytes incl. NUL)
//   int *p = null;  (TY_PTR_I32 + has_init=true, init_int=0 means null)
typedef struct {
    string  name;
    Type    type;
    bool    has_init;
    int64_t init_int;
    double  init_float;
    string  init_str;       // for TY_PTR_CHAR initialized from string literal
    int     line;
} Global;

DEFINE_VECTOR_FOR_TYPE(Global, VecGlobal)

// One module-scope `enum` constant. Resolved at emit time after locals,
// globals, and functions miss — preserves identifier shadowing.
typedef struct ProgramEnum {
    string  name;
    int64_t value;
    struct ProgramEnum *next;
} ProgramEnum;

typedef struct {
    VecFuncPtr      funcs;
    VecStructDefPtr structs;
    VecGlobal       globals;
    ProgramEnum    *enums;   // singly-linked list, module scope
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
    TC_TK_KW_CHAR,
    TC_TK_KW_TYPEDEF,
    TC_TK_KW_EXTERN,
    TC_TK_KW_SWITCH,
    TC_TK_KW_CASE,
    TC_TK_KW_DEFAULT,
    TC_TK_KW_ENUM,
    TC_TK_STRING_LIT,
    TC_TK_LPAREN, TC_TK_RPAREN,
    TC_TK_LBRACE, TC_TK_RBRACE,
    TC_TK_LBRACK, TC_TK_RBRACK,
    TC_TK_SEMI, TC_TK_COMMA, TC_TK_DOT,
    TC_TK_PLUS, TC_TK_MINUS, TC_TK_STAR, TC_TK_SLASH, TC_TK_PERCENT,
    TC_TK_LT, TC_TK_LE, TC_TK_GT, TC_TK_GE, TC_TK_EQEQ, TC_TK_NE,
    TC_TK_ASSIGN, TC_TK_BANG,
    TC_TK_AMP,
    TC_TK_PIPE, TC_TK_CARET, TC_TK_TILDE,
    TC_TK_SHL, TC_TK_SHR,
    TC_TK_AMPAMP, TC_TK_PIPEPIPE,
    TC_TK_PLUSEQ, TC_TK_MINUSEQ, TC_TK_STAREQ, TC_TK_SLASHEQ, TC_TK_PERCENTEQ,
    TC_TK_AMPEQ, TC_TK_PIPEEQ, TC_TK_CARETEQ, TC_TK_SHLEQ, TC_TK_SHREQ,
    TC_TK_PLUSPLUS, TC_TK_MINUSMINUS,
    TC_TK_QUESTION, TC_TK_COLON,
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

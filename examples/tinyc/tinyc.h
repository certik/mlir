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
//     int / float / struct / struct* / void return types.
//     `void f(...)` lowers to func.func with no results; `f(void)` is the
//     no-parameter form. `void*` is a generic pointer type (storage:
//     !llvm.ptr) implicitly convertible to/from any typed pointer; it
//     cannot be dereferenced, indexed, or used in pointer arithmetic.
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
    TY_I64,            // 64-bit signed int (long, long long, int64_t, size_t)
    TY_F32,
    TY_F64,            // 64-bit IEEE-754 double
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
    TY_VOID,           // void — only valid as a function return type
                       // or as the lone parameter spec (`f(void)`)
    TY_PTR_VOID,       // void* — generic pointer (storage: !llvm.ptr)
    TY_PTR_PTR,        // generic pointer-to-pointer (T**). Storage:
                       // !llvm.ptr. The `pointee` field carries the
                       // (single-level) inner pointer type so a deref of
                       // a T** loads the inner T*. Limited to depth 2:
                       // T*** is not supported.
    TY_VA_LIST,        // variadic argument list (`va_list`). Storage:
                       // !llvm.ptr to an !llvm.array<32 x i8> alloca, sized
                       // to cover x86_64-SysV / aarch64 layouts. Used as the
                       // operand to llvm.intr.vastart/vaend and llvm.va_arg.
} TypeKind;

typedef struct Type Type;
struct Type {
    TypeKind kind;
    int64_t  array_len;       // 1st dimension for TY_ARRAY_I32 / TY_ARRAY_STRUCT
    int64_t  array_len2;      // 2nd dimension for TY_ARRAY_I32 (0 = 1D)
    // Optional deferred array-length expression (constant integer
    // expression, may use sizeof of in-scope variables). When non-NULL,
    // the emitter folds it via ast_fold_int(scope) at the alloca site
    // and writes the result into `array_len` before allocating storage.
    // Used to support `T arr[CONST_EXPR]` where CONST_EXPR is not a bare
    // integer literal (e.g. `T arr[sizeof(x) / sizeof(x[0])]`).
    struct Expr *array_len_expr;
    string   struct_name;     // for TY_STRUCT
    // For TY_FNPTR: arena-allocated return type and parameter types.
    // Storage type is always !llvm.ptr; this signature is consulted at
    // indirect-call sites to build the func.call_indirect operand types.
    Type    *fnptr_ret;
    Type    *fnptr_params;
    int      fnptr_nparams;
    // For TY_PTR_PTR: the (single-level) pointer type that this T** points
    // at. Carries the inner pointee kind / fnptr signature / struct_name
    // so a deref can be typed correctly.
    Type    *pointee;
    // For TY_PTR_I32: when true, the pointed-at element is 64-bit wide
    // (i64). Used to map `int64_t *` / `long *` / `size_t *` to a single
    // generic pointer kind without losing the element width needed for
    // pointer arithmetic, indexing, and dereferenced-load typing.
    bool     ptr_is_i64;
    // For TY_PTR_I32: when true, the pointed-at element is f32 (float).
    // Used to map `float *` to a generic pointer kind without losing the
    // element width needed for dereferenced-load/store typing.
    bool     ptr_is_f32;
    // For TY_PTR_I32: when true, the pointed-at element is f64 (double).
    bool     ptr_is_f64;
    // For TY_ARRAY_I32: when true, the elements are 64-bit wide (i64).
    // Used to support `size_t arr[N]` / `long arr[N]` / `int64_t arr[N]`
    // without introducing a separate TY_ARRAY_I64 kind.
    bool     array_elem_is_i64;
    // For TY_ARRAY_I32: when true, the elements are 8-bit wide (i8/char).
    // Used to support `char arr[N]` without introducing a separate
    // TY_ARRAY_CHAR kind.
    bool     array_elem_is_i8;
    // For TY_I32/TY_I64: optional explicit bit width hint (0 = unspecified,
    // 8/16/32/64 = explicit width). Used by `_Generic` for typedef
    // matching on narrow-int aliases (`int8_t`, `int16_t`, ...). For
    // wasm32 we also use int_bits == 64 with kind == TY_I32 to tag
    // `long`-semantic types (storage is i32 because of the ABI, but
    // `_Generic` must keep them distinct from `int`).
    int      int_bits;
    bool     int_unsigned;
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
    EX_COMPOUND,       // (T){v0, v1, ...} — cast_type + args
    EX_VA_ARG,         // va_arg(ap, T) — lhs = ap lvalue, cast_type = T
    EX_GENERIC,        // _Generic(lhs, T1: a1, ..., default: aN)
                       // lhs = controlling expression (NOT evaluated);
                       // args[i] = result expressions; generic_types[i] =
                       // matching type for entry i (kind==TY_VOID +
                       // generic_is_default[i] flag indicates the
                       // `default:` entry).
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
    bool    is_i64;          // true iff the integer literal had L/LL suffix.
                             // When `is_i64 && !is_long_long`, the literal had
                             // a single L: it is TY_I64 on native targets but
                             // TY_I32 on wasm32 (where `long` is the size of
                             // `int`).
    bool    is_long_long;    // true iff the integer literal had an LL suffix.
                             // LL always means TY_I64 regardless of target.
    bool    is_f64;          // true iff the float literal has TY_F64 type
                             // (i.e. no `f` suffix; C's default for `1.0`).
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
    bool is_post_step;     // EX_ASSIGN that originated from a postfix `x++` /
                           // `x--`. Emit reads the old value of the lvalue,
                           // stores `old ± 1`, and yields the OLD value.
    // For EX_INDEX (lhs = array, rhs = index)
    // For EX_ADDR / EX_DEREF (lhs = inner)
    // For EX_SIZEOF / EX_CAST: cast_type is the type being asked about /
    // cast to. EX_CAST also uses lhs as the operand. EX_SIZEOF may instead
    // carry an operand expression in lhs (sizeof_is_expr=true), in which
    // case the emitter infers the type from the expression.
    Type cast_type;
    bool sizeof_is_expr;
    // For EX_COMPOUND with designated initializers: parallel to `args`.
    // compound_field_names[i].size == 0 indicates a positional entry; a
    // non-empty name selects the named field. NULL when there are no
    // designated entries at all.
    string *compound_field_names;
    // For EX_GENERIC: parallel to args[]. generic_types[i] is the type
    // at the i-th association entry; generic_default_index is the
    // index of the `default:` entry, or -1 if absent.
    Type   *generic_types;
    int     generic_default_index;
    int line;
};

typedef enum {
    ST_EXPR,           // expression-statement (e.g. f(x);)
    ST_DECL,           // <type> name [= expr];
    ST_RETURN,         // return expr;
    ST_IF,             // if (c) { then } [else { else }]
    ST_WHILE,          // while (c) { body }
    ST_DO_WHILE,       // do { body } while (c);
    ST_FOR,            // for (init; cond; step) body
    ST_BLOCK,          // { ... }
    ST_PRINT,          // print(expr);
    ST_BREAK,          // break;
    ST_CONTINUE,       // continue;
    ST_SWITCH,         // switch (e) { case 1: ... default: ... }
    ST_LABEL,          // label:
    ST_GOTO,           // goto label;
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
    // For function-local `static` variables: when non-empty, this is the
    // mangled module-scope symbol name registered in `program->globals`.
    // The emitter treats the local as an alias for that global instead
    // of allocating an `alloca` slot.
    string decl_static_global_sym;
    // ST_IF / ST_WHILE / ST_FOR / ST_SWITCH
    Expr *cond;
    VecStmtPtr then_body;
    VecStmtPtr else_body;
    bool has_else;
    VecStmtPtr while_body;
    // ST_FOR
    Stmt *for_init;
    // Zero or more for-update expressions, lowered in order at the end
    // of each iteration. Multiple entries arise from the C comma
    // operator: `for (i = 0; i < n; ++i, ++j)` parses as two entries.
    VecExprPtr for_steps;
    VecStmtPtr for_body;
    // ST_BLOCK / ST_SWITCH (body)
    VecStmtPtr block_body;
    bool block_no_scope;
    // ST_SWITCH
    VecSwitchCase switch_cases;
    // ST_LABEL / ST_GOTO
    string label_name;
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
    bool       is_variadic;  // true for `f(T, ...)` — last "parameter" was
                             // an ellipsis. Such functions are emitted as
                             // `llvm.func` (not `func.func`) so the var-arg
                             // ABI is modeled at the call site.
    bool       is_static;    // `static` at file scope: emit with private
                             // visibility/internal linkage.
    int        tu_id;        // index of the translation unit (one per
                             // tinyc_parse_into call) this came from. Used to
                             // scope `static` symbols when several source files
                             // are compiled together into one module.
    bool       is_weak;      // __attribute__((weak)): yields to a strong
                             // definition of the same symbol when merged.
    // GCC/Clang `__attribute__((...))` annotations relevant to wasm
    // codegen. Empty string means "not set". When import_module/name are
    // set the function MUST be a forward declaration; the emitter records
    // them on the resulting func op so the wasm pipeline can emit the
    // appropriate import section entry. export_name is honored on
    // function definitions and causes the function to be exported under
    // that name (used by wasm_buddy_alloc / wasm_buddy_free).
    string     wasm_import_module;
    string     wasm_import_name;
    string     wasm_export_name;
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
    bool           is_union;
} StructDef;

DEFINE_VECTOR_FOR_TYPE(StructDef*, VecStructDefPtr)

// Module-scope global variable. Restricted to literal initializers:
//   int g;          (zero)
//   int g = 42;     (init_int)
//   float f = 1.5;  (init_float)
//   char *s = "hi"; (init_str carries the decoded bytes incl. NUL)
//   int *p = null;  (TY_PTR_I32 + has_init=true, init_int=0 means null)
//   int a[3] = {1,2,3}; (init_array_data carries the packed
//                       little-endian element bytes; size equals
//                       array_len * elem_size. Empty if every
//                       element is 0/NULL — those still go through
//                       the zero-init path.)
typedef struct {
    string  name;
    Type    type;
    bool    has_init;
    bool    is_extern;      // `extern T x;` — emit external-linkage global
    bool    is_static;      // `static` at file scope: emit with internal
                            // linkage.
    int     tu_id;          // translation-unit index (see Func.tu_id).
    int64_t init_int;
    double  init_float;
    string  init_str;       // for TY_PTR_CHAR initialized from string literal
    string  init_array_data;
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

typedef struct Program {
    VecFuncPtr      funcs;
    VecStructDefPtr structs;
    VecGlobal       globals;
    ProgramEnum    *enums;   // singly-linked list, module scope
    // True when the program was parsed for the wasm32 target (set by
    // `tinyc_parse_into` from its `target_wasm32` arg). The emitter
    // uses this to size hardcoded libc-extern signatures (malloc /
    // free / printStr / tinyc_va_arg_struct) so they match the
    // wasm32-wasi ABI's 32-bit `size_t` instead of the host's 64-bit.
    bool target_wasm32;
    // Monotonic counter handing each compiled source file (one per
    // tinyc_parse_into call) a distinct translation-unit id, so `static`
    // symbols from different files merged into this Program stay file-local.
    int tu_counter;
} Program;

// ---------------- Lexer ----------------

typedef enum {
    TC_TK_EOF = 0,
    TC_TK_INT_LIT,
    TC_TK_FLOAT_LIT,
    TC_TK_IDENT,
    TC_TK_KW_INT,
    TC_TK_KW_FLOAT,
    TC_TK_KW_DOUBLE,
    TC_TK_KW_RETURN,
    TC_TK_KW_IF,
    TC_TK_KW_ELSE,
    TC_TK_KW_WHILE,
    TC_TK_KW_DO,
    TC_TK_KW_FOR,
    TC_TK_KW_BREAK,
    TC_TK_KW_CONTINUE,
    TC_TK_KW_PRINT,
    TC_TK_KW_STRUCT,
    TC_TK_KW_UNION,
    TC_TK_KW_NULL,
    TC_TK_KW_SIZEOF,
    TC_TK_KW_CHAR,
    TC_TK_KW_TYPEDEF,
    TC_TK_KW_EXTERN,
    TC_TK_KW_SWITCH,
    TC_TK_KW_CASE,
    TC_TK_KW_DEFAULT,
    TC_TK_KW_ENUM,
    TC_TK_KW_CONST,
    TC_TK_KW_VOID,
    TC_TK_KW_STATIC,
    TC_TK_KW_INLINE,
    TC_TK_KW_LONG,
    TC_TK_KW_SIGNED,
    TC_TK_KW_UNSIGNED,
    TC_TK_KW_SHORT,
    TC_TK_KW_BOOL,
    TC_TK_KW_VA_LIST,
    TC_TK_KW_GENERIC,
    TC_TK_KW_GOTO,
    TC_TK_KW_ATTRIBUTE,
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
    TC_TK_ELLIPSIS,               // ... (variadic parameter marker)
} TcTokKind;

typedef struct {
    TcTokKind kind;
    int64_t int_value;
    bool    is_i64;          // EX_INT / TC_TK_INT_LIT marked with L/LL suffix
    bool    is_long_long;    // EX_INT / TC_TK_INT_LIT marked with LL suffix
                             // (always 64-bit). When `is_i64 && !is_long_long`
                             // the literal had a single L: it is 64-bit on
                             // native targets but 32-bit on wasm32 (where
                             // `long` is `int`).
    bool    is_f64;          // TC_TK_FLOAT_LIT without `f` suffix (i.e. double)
    double  float_value;
    string text;             // interned identifier text (for IDENT)
    int line;
} TcTok;

DEFINE_VECTOR_FOR_TYPE(TcTok, VecTcTok)

VecTcTok tinyc_lex(Arena *arena, string src);

// ---------------- Preprocessor ----------------

// Run the C-style preprocessor on `path`. Returns the fully-preprocessed
// source as a single contiguous string with embedded `#line N "file"`
// directives that the lexer recognizes. `include_dirs` is a list of -I
// directories searched for `#include "..."` (after the source-relative
// search) and `#include <...>`.
string tinyc_preprocess(Arena *arena, string path,
                        string *include_dirs, size_t n_include_dirs,
                        string *defines, size_t n_defines);

// ---------------- Parser ----------------

Program *tinyc_parse(Arena *arena, VecTcTok toks);

// Multi-file: parse a single translation unit's tokens and APPEND its
// declarations (funcs, globals, structs, enums) to an existing
// Program. Cross-file deduplication is performed:
//   * funcs: forward+def and dup forwards merge; two definitions error.
//   * structs: same name with identical fields merges; mismatched errors.
//   * globals: tentative + tentative merges; tentative + initialized
//     keeps the initialized one; two initialized errors.
// Each call has its own typedef / enumerator scope (matching per-file
// preprocessor isolation): typedefs do not leak across files.
// `target_wasm32` controls the sizes of `long`, `size_t`, `intptr_t`
// and friends: on wasm32-wasi they're 32-bit; on every other supported
// host they're 64-bit (since tinyC currently runs only on
// 64-bit Linux/macOS/Windows). The flag MUST be set whenever any
// `--emit=wasm*` output is requested — otherwise tinyC will emit
// imports declared with `size_t iovs_len` as `i64`, which doesn't
// match the wasm32-wasi ABI.
// Returns the number of parse errors encountered (0 on success).
int tinyc_parse_into(Arena *arena, Program *prog, VecTcTok toks, bool target_wasm32);

// ---------------- Emitter ----------------

// Build a top-level MLIR module from `program`. The returned op handle
// is the `builtin.module` op. Uses only mlir_api.h.
MLIR_OpHandle tinyc_emit_module(MLIR_Context *ctx, Program *program);
int tinyc_last_emit_errors(void);

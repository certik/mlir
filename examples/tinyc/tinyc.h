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
//     `lhs = lhs OP rhs`, with the lvalue's address computed once
//     (Expr.is_compound), so `a[i++] += 1` increments `i` once.
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
    uint8_t  kind;            // TypeKind
    uint8_t  int_bits;
    int16_t  fnptr_nparams;
    bool     ptr_is_i64;
    bool     ptr_is_f32;
    bool     ptr_is_f64;
    bool     array_elem_is_i64;
    bool     array_elem_is_f64;
    bool     int_unsigned;
    bool     is_bool;        // `bool` / `_Bool` (int_bits is 8 like char)
    bool     is_const;       // type was declared with the `const` qualifier
    int64_t  array_len;       // 1st dimension for TY_ARRAY_I32 / TY_ARRAY_STRUCT
    int64_t  array_len2;      // 2nd dimension for TY_ARRAY_I32 (0 = 1D)
    struct Expr *array_len_expr;
    string   struct_name;     // for TY_STRUCT
    Type    *fnptr_ret;
    Type    *fnptr_params;
    Type    *pointee;
};

typedef enum {
    EX_INT,            // integer literal
    EX_FLOAT,          // float literal
    EX_STR,            // string literal (name = decoded bytes incl. NUL)
    EX_NULL,           // null pointer literal
    EX_SIZEOF,         // sizeof(<type>) / _Alignof(<type>) — uses cast_type
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
    OP_COMMA,                     // comma operator (a, b)
    OP_BAND, OP_BOR, OP_BXOR,     // bitwise & | ^
    OP_SHL, OP_SHR,               // shifts << >>
    OP_NEG, OP_NOT,               // unary - !
    OP_BNOT,                      // unary ~
} BinOp;

typedef struct Expr Expr;

DEFINE_VECTOR_FOR_TYPE(Expr*, VecExprPtr)

typedef struct {
    VecExprPtr args;
    string *compound_field_names;
    Type   *generic_types;
    int     generic_default_index;
} ExprExtra;

typedef union {
    int64_t int_value;
    double  float_value;
} ExprVal;

struct Expr {
    uint8_t kind;             // ExprKind
    uint8_t op;               // BinOp
    bool    is_i64;
    bool    is_long_long;
    bool    is_unsigned;
    bool    is_f64;
    bool    is_post_step;
    bool    is_compound;
    bool    sizeof_is_expr;
    bool    is_alignof;
    uint16_t file_id;
    int line;
    ExprVal val;              // int_value / float_value
    string  name;             // EX_VAR / EX_STR / EX_FIELD / EX_CALL (callee)
    Expr   *lhs;
    Expr   *rhs;              // EX_BIN / EX_TERNARY (then) / EX_ASSIGN (rhs)
    Expr   *lvalue;           // EX_ASSIGN (lhs) / EX_TERNARY (else)
    Type   *cast_type;
    ExprExtra *extra;         // EX_CALL / EX_COMPOUND / EX_GENERIC metadata
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
    string  label;
} SwitchCase;

DEFINE_VECTOR_FOR_TYPE(SwitchCase, VecSwitchCase)

typedef struct {
    // For function-local `static` variables: mangled module-scope symbol name.
    string        decl_static_global_sym;
    // ST_IF (only allocated when `else` is present)
    VecStmtPtr    else_body;
    // ST_FOR
    Stmt         *for_init;
    VecExprPtr    for_steps;
    // ST_SWITCH
    VecSwitchCase switch_cases;
} StmtExtra;

struct Stmt {
    uint8_t kind;            // StmtKind
    bool decl_is_static;
    bool has_else;
    bool block_no_scope;
    uint16_t file_id;
    int line;
    // ST_EXPR / ST_RETURN / ST_DECL (init) / ST_IF / ST_WHILE / ST_DO_WHILE / ST_FOR / ST_SWITCH (cond)
    Expr *expr;
    // ST_DECL (decl_name) / ST_LABEL / ST_GOTO (label_name)
    string name;
    // ST_IF (then) / ST_WHILE / ST_DO_WHILE / ST_FOR / ST_BLOCK / ST_SWITCH
    VecStmtPtr body;
    // ST_DECL only (arena-allocated)
    Type *decl_type;
    // ST_IF(else) / ST_FOR / ST_SWITCH / ST_DECL(static) rare metadata
    StmtExtra *extra;
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
    uint16_t   file_id;
    int        line;
} Func;

DEFINE_VECTOR_FOR_TYPE(Func*, VecFuncPtr)

typedef struct {
    string  name;
    Type    type;        // field type.
    uint8_t bit_width;   // bit-field width in bits (0 for normal fields or `: 0`)
    bool    is_bitfield; // true for any `: N` bit-field (including `: 0`)
    uint8_t bit_offset;  // bit offset within its backing storage unit
    int     llvm_field_idx; // backing LLVM struct field index
} StructField;

DEFINE_VECTOR_FOR_TYPE(StructField, VecStructField)

typedef struct StructDef {
    string         name;
    VecStructField fields;
    uint16_t       file_id;
    int            line;
    bool           is_union;
    bool           layout_computed;
} StructDef;

DEFINE_VECTOR_FOR_TYPE(StructDef*, VecStructDefPtr)

// Module-scope global variable. Restricted to literal initializers:
//   int g;          (zero)
//   int g = 42;     (init_int)
//   float f = 1.5;  (init_float)
//   char *s = "hi"; (init_str carries the decoded bytes incl. NUL)
//   int *p = null;  (TY_PTR_I32 + has_init=true, init_int=0 means null)
//   char s[8] = "hi"; (init_str; the emitter pads it to the array)
//   struct S g = {1, 2}; (init_expr: the EX_COMPOUND, folded by the
//                       emitter into the global's constant data)
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
    uint32_t address_space; // LLVM pointer AS for addressof/load/store (0=default)
    bool    is_const;       // `const` at file scope: emit as llvm constant
    int64_t init_int;
    double  init_float;
    string  init_str;       // for TY_PTR_CHAR initialized from string literal
    string  init_array_data;
    struct Expr *init_expr; // `{...}` initializer of a struct / struct array
    uint16_t file_id;
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
    // free / tinyc_va_arg_struct) so they match the
    // wasm32-wasi ABI's 32-bit `size_t` instead of the host's 64-bit.
    bool target_wasm32;
    // When set, variadic functions that are NOT defined in this TU keep the
    // target's native C varargs ABI (for linking against host libc printf on
    // the native-llc / ELF test path). When clear (the default), EVERY
    // variadic function uses tinyC's portable 8-byte cursor va_list — correct
    // for wasm/macho and for separate-compilation builds where a tinyC-defined
    // variadic function (e.g. corec-stdlib printf) is extern in the caller's TU.
    bool host_varargs;
    // Interned source file names (1-based file_id indexes source_files[file_id - 1]).
    string *source_files;
    size_t  n_source_files;
    size_t  source_files_cap;
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
    TC_TK_KW_STRUCT,
    TC_TK_KW_UNION,
    TC_TK_KW_NULL,
    TC_TK_KW_SIZEOF,
    TC_TK_KW_ALIGNOF,             // _Alignof: parsed like sizeof (EX_SIZEOF)
    TC_TK_KW_CHAR,
    TC_TK_KW_TYPEDEF,
    TC_TK_KW_EXTERN,
    TC_TK_KW_SWITCH,
    TC_TK_KW_CASE,
    TC_TK_KW_DEFAULT,
    TC_TK_KW_ENUM,
    TC_TK_KW_CONST,
    TC_TK_KW_VOLATILE,
    TC_TK_KW_RESTRICT,
    TC_TK_KW_REGISTER,
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
    TC_TK_KW_DECLSPEC,
    TC_TK_KW_CALLCONV,
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

#define TC_TOK_F_I64       1u
#define TC_TOK_F_LONG_LONG 2u
#define TC_TOK_F_UNSIGNED  4u
#define TC_TOK_F_F64       8u

typedef union {
    uint64_t size;
    int64_t  int_value;
    double   float_value;
} TcTokData;

typedef struct {
    uint8_t   kind;          // TcTokKind
    uint8_t   flags;         // TC_TOK_F_*
    uint16_t  file_id;       // 1-based index into source_files (0 = unknown)
    int       line;
    char     *text_str;      // identifier / string literal bytes
    TcTokData d;             // text size / int_value / float_value
} TcTok;

static inline string tc_tok_text(TcTok t) {
    string s;
    s.str = t.text_str;
    s.size = t.d.size;
    return s;
}

static inline void tc_tok_set_text(TcTok *t, string s) {
    t->text_str = s.str;
    t->d.size = s.size;
}

DEFINE_VECTOR_FOR_TYPE(TcTok, VecTcTok)

VecTcTok tinyc_lex(Arena *arena, string src);
void     tinyc_set_last_lex_files(string *files, size_t n);
string  *tinyc_last_lex_files(size_t *out_n);
TcTokKind tinyc_keyword_or_ident(string s);
TcTok tinyc_lex_number(string text, int line);
TcTok tinyc_lex_string(Arena *arena, string raw, int line);
TcTok tinyc_lex_char(string raw, int line);

// Integer constant `v` converted to the C type `t` when that is a narrow
// integer: char / short are truncated and re-extended by signedness, bool
// becomes 0 / 1. Other types return `v` unchanged.
int64_t tinyc_narrow_const(int64_t v, Type t);

// Storage size in bytes of an integer of type `t`, or of one element of the
// integer array type `t` (TY_ARRAY_I32): 1 for char / bool, 2 for short, 8
// for a 64-bit array element, else 4.
int tinyc_int_size(const Type *t);

// ---------------- Preprocessor ----------------

// Run the C-style preprocessor on `path`. Returns the fully-preprocessed
// source as a single contiguous string with embedded `#line N "file"`
// directives that the lexer recognizes. `include_dirs` is a list of -I
// directories searched for `#include "..."` (after the source-relative
// search) and `#include <...>`.
string tinyc_preprocess(Arena *arena, string path,
                        string *include_dirs, size_t n_include_dirs,
                        string *defines, size_t n_defines);

// Fused preprocess + lex: emits TcTok directly without serializing to text
// and re-lexing. Equivalent to tinyc_lex(arena, tinyc_preprocess(...)).
VecTcTok tinyc_preprocess_lex(Arena *arena, string path,
                              string *include_dirs, size_t n_include_dirs,
                              string *defines, size_t n_defines);

// Number of errors (missing #include, #error, malformed directives, ...)
// the most recent tinyc_preprocess() call reported. The returned text is
// incomplete when this is non-zero, so callers must not compile it.
int tinyc_last_preprocess_errors(void);

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

// ---------------- Constant folding ----------------

// A folded integer constant: its value, held sign- or zero-extended from
// its width by its signedness, and its C type (int / unsigned / 64-bit
// variants).
typedef struct {
    int64_t v;
    bool is_i64;
    bool is_unsigned;
} FoldVal;

// Folds the leaves tinyc_fold does not know itself (`sizeof` in the
// emitter, enumerators in the parser); returns false to refuse. `lenient`:
// the leaf is in an operand that is not evaluated, so only its type
// matters (an unevaluated `sizeof` folds to a 0 of type size_t).
typedef bool (*FoldLeafFn)(void *ctx, Expr *ex, bool lenient, FoldVal *out);

// Fold a constant integer expression with C's typing rules (emit.c).
bool tinyc_fold(Expr *ex, FoldLeafFn leaf, void *ctx, FoldVal *out);

// ---------------- Emitter ----------------

// Build a top-level MLIR module from `program`. The returned op handle
// is the `builtin.module` op. Uses only mlir_api.h.
MLIR_OpHandle tinyc_emit_module(MLIR_Context *ctx, Program *program);
int tinyc_last_emit_errors(void);

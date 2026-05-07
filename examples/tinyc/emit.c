// tinyC -> MLIR emitter (LLVM-dialect memory model). Uses ONLY mlir_api.h
// plus corec primitives. No upstream MLIR headers are included from this
// directory.
//
// Memory model:
//   - Every local is one !llvm.ptr produced by `llvm.alloca <count> x ELEM`.
//   - Scalar local: ELEM = i32 / f32; load/store with llvm.load / llvm.store.
//   - Struct local: ELEM = !llvm.struct<"S", (...)>; field access via
//     llvm.getelementptr + llvm.load / llvm.store.
//   - int[N] local: ELEM = !llvm.array<N x i32>; indexed via GEP [0, i].
//   - struct[N] local: ELEM = !llvm.array<N x !llvm.struct<...>>; field via
//     GEP [0, i, field_idx].
//   - int* local: ELEM = !llvm.ptr; *p loads p first to get inner ptr.
//   - struct* local: ELEM = !llvm.ptr; p->f loads first then GEP+load.
//
// ABI:
//   - int / float param: scalar.
//   - struct param: caller allocates a fresh copy and passes !llvm.ptr; the
//     callee binds its sym->addr to that block-arg directly (caller-byval).
//   - struct* param: !llvm.ptr; callee alloca's a local !llvm.ptr slot and
//     stores the block-arg into it.
//   - struct return: function returns void and takes a hidden first
//     !llvm.ptr argument (sret) into which the callee writes.
//
// Control flow is emitted in cf-form (cf.br / cf.cond_br) for if/while/for
// and `break`/`continue`/early-return. Short-circuit && / || still use
// scf.if-with-result inside the expression itself; SCFToCF runs as part of
// MLIR_LowerToLLVMDialect.

#include "tinyc.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <stdlib.h>
#include <base/arena.h>
#include <base/format.h>
#include <base/io.h>
#include <base/string.h>

// ----- symbol table -----

typedef struct Sym {
    string name;
    Type   type;                // declared type
    MLIR_ValueHandle addr;      // !llvm.ptr to this local's storage
    StructDef *sdef;            // for TY_STRUCT / TY_PTR_STRUCT / TY_ARRAY_STRUCT
    bool   is_global;           // if true, addr is unused; reload via addressof
    string global_sym;          // sym_name of the llvm.mlir.global (for globals)
    struct Sym *next;
} Sym;

typedef struct Scope {
    Sym *head;
    struct Scope *parent;
} Scope;

static Sym *scope_lookup(Scope *s, string name) {
    for (Scope *cur = s; cur; cur = cur->parent) {
        for (Sym *sy = cur->head; sy; sy = sy->next) {
            if (str_eq(sy->name, name)) return sy;
        }
    }
    return NULL;
}

// ----- emit context -----

typedef struct LoopCtx {
    MLIR_BlockHandle continue_block;
    MLIR_BlockHandle break_block;
    struct LoopCtx *parent;
} LoopCtx;

typedef struct {
    Type        type;          // declared slot type
    StructDef  *sdef;          // resolved if struct/struct*/struct[]
    // ABI: every slot maps to exactly ONE flat MLIR operand/result type.
    // For struct values that's !llvm.ptr (caller-byval); for pointer/scalar
    // it's the natural type. For sret returns the callee's flat return is
    // 0 and a hidden !llvm.ptr is prepended to the parameters.
} SlotInfo;

typedef struct FuncSig {
    string         name;
    Func          *func;
    SlotInfo       ret;
    SlotInfo      *params;
    size_t         n_params;
    bool           sret;             // struct return -> hidden first ptr param
    bool           is_variadic;      // C `f(T, ...)` — emitted as `llvm.func`
                                     // and called via `llvm.call` with a
                                     // `var_callee_type` attribute.
    MLIR_TypeHandle *flat_in_tys;    // including sret arg if sret
    size_t         n_flat_in;
    MLIR_TypeHandle *flat_out_tys;   // 0 if void/sret, 1 otherwise
    size_t         n_flat_out;
    MLIR_TypeHandle fn_ty;           // FunctionType (non-variadic) or
                                     // LLVMFunctionType (variadic).
    MLIR_TypeHandle llvm_fn_ty;      // LLVMFunctionType — only set for
                                     // variadic; used as `var_callee_type`.
} FuncSig;

typedef struct StructTypeEntry {
    StructDef       *sd;
    MLIR_TypeHandle  ty;
} StructTypeEntry;

typedef struct StringEntry {
    string  bytes;
    string  sym;        // global symbol name like ".str.0"
} StringEntry;

typedef struct {
    MLIR_Context *ctx;
    Arena *arena;
    MLIR_TypeHandle i32;
    MLIR_TypeHandle i1;
    MLIR_TypeHandle i64;
    MLIR_TypeHandle i8;
    MLIR_TypeHandle f32;
    MLIR_TypeHandle index;
    MLIR_TypeHandle ptr;             // !llvm.ptr
    MLIR_BlockHandle cur_block;
    bool terminated;
    MLIR_RegionHandle func_region;
    MLIR_LocationHandle loc;
    int next_ssa;
    LoopCtx *loops;
    Program *program;
    FuncSig *sigs;
    size_t   n_sigs;
    FuncSig *cur_sig;
    MLIR_ValueHandle cur_sret_ptr;   // for struct-returning functions
    StructTypeEntry *struct_types;
    size_t           n_struct_types;
    StringEntry     *strings;        // dedup'd string-literal pool
    size_t           n_strings;
    size_t           cap_strings;
    MLIR_BlockHandle module_block;
    Sym             *globals;        // module-scope symbols
    bool             use_print_str;  // emit @printStr extern decl
    bool             need_va_arg_helpers;  // emit tinyc_va_arg_* externs
    int              cur_line;       // last AST node line entered; used by
                                     // EMIT_ERR for diagnostic line numbers.
} E;

// All emit-time diagnostics route through this macro so they carry a line
// number. `msg` must be a C string literal (it is concatenated with the
// "tinyc emit error at line {}: " prefix at preprocess time and wrapped in
// str_lit). The line is taken from e->cur_line, which is updated at the
// entry of emit_expr / emit_lvalue / emit_stmt / emit_func / per-struct
// pre-pass loops.
#define EMIT_ERR(e, msg, ...) \
    println(str_lit("tinyc emit error at line {}: " msg), \
            (int64_t)((e)->cur_line), ##__VA_ARGS__)

static StructDef *find_struct(E *e, string name) {
    if (!e->program) return NULL;
    for (size_t i = 0; i < e->program->structs.size; i++) {
        StructDef *sd = e->program->structs.data[i];
        if (str_eq(sd->name, name)) return sd;
    }
    return NULL;
}

static MLIR_TypeHandle find_struct_type(E *e, StructDef *sd) {
    for (size_t i = 0; i < e->n_struct_types; i++) {
        if (e->struct_types[i].sd == sd) return e->struct_types[i].ty;
    }
    return MLIR_INVALID_HANDLE;
}

static int struct_field_index(StructDef *sd, string name) {
    for (size_t i = 0; i < sd->fields.size; i++) {
        if (str_eq(sd->fields.data[i].name, name)) return (int)i;
    }
    return -1;
}

static FuncSig *find_sig(E *e, string name) {
    for (size_t i = 0; i < e->n_sigs; i++) {
        if (str_eq(e->sigs[i].name, name)) return &e->sigs[i];
    }
    return NULL;
}

// Emit-time enumerator lookup. Resolved AFTER locals/globals/functions
// so that an inner-scope variable with the same name correctly shadows
// the enum constant (e.g. `enum {A=1}; int f(){ int A=2; return A; }`
// must return 2, not 1).
static bool find_enum(E *e, string name, int64_t *out_value) {
    if (!e->program) return false;
    for (ProgramEnum *pe = e->program->enums; pe; pe = pe->next) {
        if (str_eq(pe->name, name)) { *out_value = pe->value; return true; }
    }
    return false;
}

static MLIR_TypeHandle scalar_mlir_type(E *e, TypeKind k) {
    if (k == TY_F32) return e->f32;
    if (k == TY_I64) return e->i64;
    if (k == TY_PTR_STRUCT || k == TY_PTR_I32 || k == TY_PTR_CHAR ||
        k == TY_PTR_VOID || k == TY_FNPTR || k == TY_PTR_PTR ||
        k == TY_VA_LIST) return e->ptr;
    return e->i32;
}

static MLIR_LocationHandle eloc(E *e, int line) {
    (void)line;
    return e->loc;
}

static string ssa_name(E *e) {
    int n = e->next_ssa++;
    return format(e->arena, str_lit("%{}"), (int64_t)n);
}

static MLIR_BlockHandle new_cfg_block(E *e) {
    MLIR_BlockHandle b = MLIR_CreateBlock(e->ctx);
    MLIR_AppendRegionBlock(e->ctx, e->func_region, b);
    return b;
}

static MLIR_OpHandle emit_op(E *e,
                             MLIR_OpType type,
                             string opname,
                             MLIR_TypeHandle *result_types, size_t n_result_types,
                             MLIR_ValueHandle *results, size_t n_results,
                             MLIR_ValueHandle *operands, size_t n_operands,
                             MLIR_AttributeHandle *attrs, size_t n_attrs,
                             MLIR_RegionHandle *regions, size_t n_regions) {
    MLIR_OpHandle op = MLIR_CreateOp(e->ctx, type, opname,
                                     attrs, n_attrs,
                                     result_types, n_result_types,
                                     results, n_results,
                                     operands, n_operands,
                                     regions, n_regions,
                                     eloc(e, 0), MLIR_INVALID_HANDLE,
                                     str_lit(""), -1);
    if (e->cur_block) MLIR_AppendBlockOp(e->ctx, e->cur_block, op);
    return op;
}

// Emit a function-return op. The right op depends on the surrounding
// function dialect: variadic functions are emitted as `llvm.func`, all
// others as `func.func`. Number of operands is 0 (void/sret returns) or 1
// (scalar/pointer return). LLVMReturnOp accepts at most one operand.
static void emit_return_op(E *e, MLIR_ValueHandle *ops, size_t n_ops) {
    if (e->cur_sig && e->cur_sig->is_variadic) {
        emit_op(e, OP_TYPE_LLVM_RETURN, str_lit("llvm.return"),
                NULL, 0, NULL, 0, ops, n_ops, NULL, 0, NULL, 0);
    } else {
        emit_op(e, OP_TYPE_FUNC_RETURN, str_lit("func.return"),
                NULL, 0, NULL, 0, ops, n_ops, NULL, 0, NULL, 0);
    }
}

static void emit_branch(E *e, MLIR_BlockHandle target) {
    if (e->terminated) return;
    MLIR_BlockHandle *succs = arena_new_array(e->arena, MLIR_BlockHandle, 1);
    succs[0] = target;
    MLIR_ValueHandle **sops = arena_new_array(e->arena, MLIR_ValueHandle *, 1);
    sops[0] = NULL;
    size_t *snums = arena_new_array(e->arena, size_t, 1);
    snums[0] = 0;
    MLIR_OpHandle op = MLIR_CreateOpWithSuccessors(
        e->ctx, OP_TYPE_CF_BR, str_lit("cf.br"),
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0,
        succs, 1, sops, snums,
        eloc(e, 0), MLIR_INVALID_HANDLE, str_lit(""), -1);
    if (e->cur_block) MLIR_AppendBlockOp(e->ctx, e->cur_block, op);
    e->terminated = true;
}

static void emit_cond_branch(E *e, MLIR_ValueHandle cond,
                             MLIR_BlockHandle true_b, MLIR_BlockHandle false_b) {
    if (e->terminated) return;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1);
    ops[0] = cond;
    MLIR_BlockHandle *succs = arena_new_array(e->arena, MLIR_BlockHandle, 2);
    succs[0] = true_b; succs[1] = false_b;
    MLIR_ValueHandle **sops = arena_new_array(e->arena, MLIR_ValueHandle *, 2);
    sops[0] = NULL; sops[1] = NULL;
    size_t *snums = arena_new_array(e->arena, size_t, 2);
    snums[0] = 0; snums[1] = 0;
    MLIR_OpHandle op = MLIR_CreateOpWithSuccessors(
        e->ctx, OP_TYPE_CF_COND_BR, str_lit("cf.cond_br"),
        NULL, 0, NULL, 0, NULL, 0, ops, 1, NULL, 0,
        succs, 2, sops, snums,
        eloc(e, 0), MLIR_INVALID_HANDLE, str_lit(""), -1);
    if (e->cur_block) MLIR_AppendBlockOp(e->ctx, e->cur_block, op);
    e->terminated = true;
}

// ----- constants -----

static MLIR_ValueHandle emit_const_i32(E *e, int64_t v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i32, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i32;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_AttributeHandle val = MLIR_CreateAttributeInteger(e->ctx, str_lit("value"), v, e->i32);
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = val;
    emit_op(e, OP_TYPE_ARITH_CONSTANT, str_lit("arith.constant"),
            rts, 1, rs, 1, NULL, 0, as, 1, NULL, 0);
    return r;
}

static MLIR_ValueHandle emit_const_f32(E *e, double v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->f32, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->f32;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_AttributeHandle val = MLIR_CreateAttributeFloat(e->ctx, str_lit("value"), v, e->f32);
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = val;
    emit_op(e, OP_TYPE_ARITH_CONSTANT, str_lit("arith.constant"),
            rts, 1, rs, 1, NULL, 0, as, 1, NULL, 0);
    return r;
}

static MLIR_ValueHandle emit_const_i64(E *e, int64_t v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i64, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i64;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_AttributeHandle val = MLIR_CreateAttributeInteger(e->ctx, str_lit("value"), v, e->i64);
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = val;
    emit_op(e, OP_TYPE_ARITH_CONSTANT, str_lit("arith.constant"),
            rts, 1, rs, 1, NULL, 0, as, 1, NULL, 0);
    return r;
}

static MLIR_ValueHandle emit_binop(E *e, MLIR_OpType type, string name,
                                   MLIR_TypeHandle rt,
                                   MLIR_ValueHandle a, MLIR_ValueHandle b) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  rt, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = rt;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 2); ops[0] = a; ops[1] = b;
    emit_op(e, type, name, rts, 1, rs, 1, ops, 2, NULL, 0, NULL, 0);
    return r;
}

static MLIR_ValueHandle emit_cmpi(E *e, int64_t predicate,
                                  MLIR_ValueHandle a, MLIR_ValueHandle b) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i1, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i1;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 2); ops[0] = a; ops[1] = b;
    MLIR_AttributeHandle pred = MLIR_CreateAttributeInteger(e->ctx, str_lit("predicate"),
                                                            predicate,
                                                            MLIR_CreateTypeInteger(e->ctx, 64, true));
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = pred;
    emit_op(e, OP_TYPE_ARITH_CMPI, str_lit("arith.cmpi"),
            rts, 1, rs, 1, ops, 2, as, 1, NULL, 0);
    return r;
}

static MLIR_ValueHandle emit_cmpf(E *e, int64_t predicate,
                                  MLIR_ValueHandle a, MLIR_ValueHandle b) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i1, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i1;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 2); ops[0] = a; ops[1] = b;
    MLIR_AttributeHandle pred = MLIR_CreateAttributeInteger(e->ctx, str_lit("predicate"),
                                                            predicate,
                                                            MLIR_CreateTypeInteger(e->ctx, 64, true));
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = pred;
    emit_op(e, OP_TYPE_ARITH_CMPF, str_lit("arith.cmpf"),
            rts, 1, rs, 1, ops, 2, as, 1, NULL, 0);
    return r;
}

static MLIR_ValueHandle emit_icmp_ptr(E *e, int64_t predicate,
                                      MLIR_ValueHandle a, MLIR_ValueHandle b) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i1, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i1;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 2); ops[0] = a; ops[1] = b;
    MLIR_AttributeHandle pred = MLIR_CreateAttributeInteger(e->ctx, str_lit("predicate"),
                                                            predicate,
                                                            MLIR_CreateTypeInteger(e->ctx, 64, true));
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = pred;
    emit_op(e, OP_TYPE_LLVM_ICMP, str_lit("llvm.icmp"),
            rts, 1, rs, 1, ops, 2, as, 1, NULL, 0);
    return r;
}

static MLIR_ValueHandle emit_null_ptr(E *e) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->ptr, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->ptr;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    emit_op(e, OP_TYPE_LLVM_MLIR_ZERO, str_lit("llvm.mlir.zero"),
            rts, 1, rs, 1, NULL, 0, NULL, 0, NULL, 0);
    return r;
}

static MLIR_ValueHandle emit_extsi_i32_to_i64(E *e, MLIR_ValueHandle v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i64, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i64;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v;
    emit_op(e, OP_TYPE_ARITH_EXTSI, str_lit("arith.extsi"),
            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
    return r;
}

// Truncate i64 -> i32 (arith.trunci).
static MLIR_ValueHandle emit_trunci_i64_to_i32(E *e, MLIR_ValueHandle v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i32, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i32;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v;
    emit_op(e, OP_TYPE_ARITH_TRUNCI, str_lit("arith.trunci"),
            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
    return r;
}

// Emit `llvm.mlir.addressof @<sym>` -> !llvm.ptr.
static MLIR_ValueHandle emit_addressof(E *e, string sym) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->ptr, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->ptr;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_AttributeHandle a = MLIR_CreateAttributeSymbolRef(
        e->ctx, str_lit("global_name"), sym);
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = a;
    emit_op(e, OP_TYPE_LLVM_MLIR_ADDRESSOF, str_lit("llvm.mlir.addressof"),
            rts, 1, rs, 1, NULL, 0, as, 1, NULL, 0);
    return r;
}

// Look up (or create) a string-literal global; return its sym_name.
static string intern_string(E *e, string bytes) {
    for (size_t i = 0; i < e->n_strings; i++) {
        if (e->strings[i].bytes.size == bytes.size &&
            memcmp(e->strings[i].bytes.str, bytes.str, bytes.size) == 0) {
            return e->strings[i].sym;
        }
    }
    if (e->n_strings == e->cap_strings) {
        size_t nc = e->cap_strings ? e->cap_strings * 2 : 8;
        StringEntry *na = arena_new_array(e->arena, StringEntry, nc);
        for (size_t i = 0; i < e->n_strings; i++) na[i] = e->strings[i];
        e->strings = na; e->cap_strings = nc;
    }
    string sym = format(e->arena, str_lit(".str.{}"), (int64_t)e->n_strings);
    e->strings[e->n_strings].bytes = bytes;
    e->strings[e->n_strings].sym   = sym;
    e->n_strings++;
    return sym;
}

// Address of a symbol: either the local alloca slot or a freshly-emitted
// addressof for a module-level global.
static MLIR_ValueHandle sym_addr(E *e, Sym *s) {
    if (s->is_global) return emit_addressof(e, s->global_sym);
    return s->addr;
}

// Look up a name as a global. Returns NULL if not found.
static Sym *global_lookup(E *e, string name) {
    for (Sym *s = e->globals; s; s = s->next) {
        if (str_eq(s->name, name)) return s;
    }
    return NULL;
}

// Combined lookup: locals/scope chain first, then module globals.
static Sym *lookup(E *e, Scope *sc, string name) {
    Sym *s = scope_lookup(sc, name);
    if (s) return s;
    return global_lookup(e, name);
}

// ----- LLVM-dialect storage primitives -----

static MLIR_ValueHandle emit_alloca(E *e, MLIR_TypeHandle elem_ty) {
    MLIR_ValueHandle one = emit_const_i64(e, 1);
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->ptr, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->ptr;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = one;
    MLIR_AttributeHandle elem_attr = MLIR_CreateAttributeType(e->ctx, str_lit("elem_type"), elem_ty);
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = elem_attr;
    emit_op(e, OP_TYPE_LLVM_ALLOCA, str_lit("llvm.alloca"),
            rts, 1, rs, 1, ops, 1, as, 1, NULL, 0);
    return r;
}

static void emit_store_v(E *e, MLIR_ValueHandle val, MLIR_ValueHandle ptr) {
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 2);
    ops[0] = val; ops[1] = ptr;
    emit_op(e, OP_TYPE_LLVM_STORE, str_lit("llvm.store"),
            NULL, 0, NULL, 0, ops, 2, NULL, 0, NULL, 0);
}

static MLIR_ValueHandle emit_load_v(E *e, MLIR_ValueHandle ptr, MLIR_TypeHandle elem_ty) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  elem_ty, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = elem_ty;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = ptr;
    emit_op(e, OP_TYPE_LLVM_LOAD, str_lit("llvm.load"),
            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
    return r;
}

// llvm.getelementptr base, raw_idx, dyn_idx... -> !llvm.ptr.
// raw_idx[k] is either a constant index or kDynamicIndex (= INT32_MIN) for
// each dynamic slot. dyn_idx[] gives the SSA values for the dynamic slots
// in left-to-right order.
static const int32_t LLVM_GEP_DYN = (int32_t)0x80000000;

static MLIR_ValueHandle emit_gep(E *e,
        MLIR_ValueHandle base,
        MLIR_TypeHandle source_elem_ty,
        const int32_t *raw_idx, size_t n_raw,
        MLIR_ValueHandle *dyn_idx, size_t n_dyn) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->ptr, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->ptr;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    size_t n_ops = 1 + n_dyn;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, n_ops);
    ops[0] = base;
    for (size_t i = 0; i < n_dyn; i++) ops[1 + i] = dyn_idx[i];
    MLIR_AttributeHandle elem_attr = MLIR_CreateAttributeType(e->ctx, str_lit("elem_type"), source_elem_ty);
    MLIR_AttributeHandle raw_attr = MLIR_CreateAttributeDenseI32Array(
        e->ctx, str_lit("rawConstantIndices"), raw_idx, n_raw);
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 2);
    as[0] = raw_attr; as[1] = elem_attr;
    emit_op(e, OP_TYPE_LLVM_GEP, str_lit("llvm.getelementptr"),
            rts, 1, rs, 1, ops, n_ops, as, 2, NULL, 0);
    return r;
}

static MLIR_ValueHandle emit_select(E *e, MLIR_TypeHandle rt, MLIR_ValueHandle c,
                                    MLIR_ValueHandle t, MLIR_ValueHandle f) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  rt, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = rt;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 3);
    ops[0] = c; ops[1] = t; ops[2] = f;
    emit_op(e, OP_TYPE_ARITH_SELECT, str_lit("arith.select"),
            rts, 1, rs, 1, ops, 3, NULL, 0, NULL, 0);
    return r;
}

static MLIR_ValueHandle emit_sitofp(E *e, MLIR_ValueHandle i32_val) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->f32, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->f32;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = i32_val;
    emit_op(e, OP_TYPE_ARITH_SITOFP, str_lit("arith.sitofp"),
            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
    return r;
}

static int64_t cmpi_pred_for(BinOp op) {
    switch (op) {
        case OP_EQ: return 0; case OP_NE: return 1;
        case OP_LT: return 2; case OP_LE: return 3;
        case OP_GT: return 4; case OP_GE: return 5;
        default:    return 0;
    }
}

static int64_t cmpf_pred_for(BinOp op) {
    switch (op) {
        case OP_EQ: return 1; case OP_NE: return 6;
        case OP_LT: return 4; case OP_LE: return 5;
        case OP_GT: return 2; case OP_GE: return 3;
        default:    return 1;
    }
}

// ----- expression / statement emission -----

typedef struct {
    MLIR_ValueHandle val;
    bool is_float;
    bool is_i64;       // val has MLIR type i64 (TY_I64)
    bool is_ptr;       // val is a !llvm.ptr (struct pointer, int*, char*, ...)
    bool is_str;       // val is a !llvm.ptr to char data (string)
    StructDef *sdef;   // when is_ptr, the StructDef the pointer targets (NULL for null)
    MLIR_TypeHandle ptr_elem;  // pointee element type for non-struct pointers
                               // (INVALID otherwise). Used by pointer arithmetic.
    Type *fnptr_ty;    // when val is a function pointer, its TY_FNPTR signature
                       // (return + parameter types). NULL otherwise.
    bool is_void_ptr;  // val is a `void*` — dereference / indexing /
                       // pointer arithmetic are forbidden on it.
} EVal;

static EVal emit_expr(E *e, Scope *sc, Expr *ex);
static int64_t type_size(E *e, Type t);
static int64_t type_align(E *e, Type t);
static Type infer_expr_type(E *e, Scope *sc, Expr *ex);
static void emit_struct_zero(E *e, MLIR_ValueHandle base, MLIR_TypeHandle source_elem,
                             StructDef *sd, int32_t *prefix, size_t n_prefix);

static MLIR_ValueHandle emit_expr_i32(E *e, Scope *sc, Expr *ex) {
    EVal v = emit_expr(e, sc, ex);
    if (v.is_float) {
        MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                      e->i32, ssa_name(e), eloc(e, 0));
        MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i32;
        MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
        MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v.val;
        emit_op(e, OP_TYPE_ARITH_FPTOSI, str_lit("arith.fptosi"),
                rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
        return r;
    }
    return v.val;
}

static MLIR_ValueHandle coerce_eval(E *e, EVal v, MLIR_TypeHandle want) {
    if (want == e->f32 && !v.is_float) return emit_sitofp(e, v.val);
    if (want == e->i32 && v.is_float) {
        MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                      e->i32, ssa_name(e), eloc(e, 0));
        MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i32;
        MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
        MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v.val;
        emit_op(e, OP_TYPE_ARITH_FPTOSI, str_lit("arith.fptosi"),
                rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
        return r;
    }
    if (want == e->i64) {
        if (v.is_i64) return v.val;
        if (v.is_ptr) return v.val;
        if (v.is_float) {
            // float -> i32 -> i64 (no direct fptosi-to-i64 helper here).
            EVal as_i32 = (EVal){.val = coerce_eval(e, v, e->i32)};
            return emit_extsi_i32_to_i64(e, as_i32.val);
        }
        return emit_extsi_i32_to_i64(e, v.val);
    }
    if (want == e->i32 && v.is_i64) {
        return emit_trunci_i64_to_i32(e, v.val);
    }
    return v.val;
}

static MLIR_ValueHandle emit_to_bool_i1(E *e, EVal v) {
    if (v.is_float) {
        MLIR_ValueHandle z = emit_const_f32(e, 0.0);
        return emit_cmpf(e, /*one*/ 6, v.val, z);
    }
    MLIR_ValueHandle z = emit_const_i32(e, 0);
    return emit_cmpi(e, /*ne*/ 1, v.val, z);
}

static MLIR_ValueHandle bool_to_i32(E *e, MLIR_ValueHandle b) {
    MLIR_ValueHandle one  = emit_const_i32(e, 1);
    MLIR_ValueHandle zero = emit_const_i32(e, 0);
    return emit_select(e, e->i32, b, one, zero);
}

// ----- LVal: a generalized lvalue described as a GEP-and-load/store target.
//
// The lvalue points at a leaf scalar (i32 or f32). It is described as either
// a direct !llvm.ptr to that scalar (n_const_path == 0 && dyn_index INVALID
// — typical for a simple int/float local or a *p result), or as a base
// !llvm.ptr together with a GEP descent (`source_elem` = type pointed to by
// base, `const_path` / `dyn_index` describe the indices). Resolution emits
// a single GEP and then load/store.

typedef struct {
    MLIR_ValueHandle base_ptr;
    MLIR_TypeHandle  source_elem;     // INVALID if no GEP needed
    int32_t         *const_path;
    size_t           n_const_path;
    MLIR_ValueHandle dyn_index;       // INVALID for purely-static GEP
    MLIR_TypeHandle  elem_ty;         // i32 or f32
} LVal;

static MLIR_ValueHandle lval_address(E *e, LVal lv) {
    if (lv.n_const_path == 0 && lv.dyn_index == MLIR_INVALID_HANDLE) {
        return lv.base_ptr;
    }
    MLIR_ValueHandle dyn[1];
    size_t n_dyn = 0;
    if (lv.dyn_index != MLIR_INVALID_HANDLE) {
        dyn[0] = lv.dyn_index;
        n_dyn = 1;
    }
    return emit_gep(e, lv.base_ptr, lv.source_elem,
                    lv.const_path, lv.n_const_path, dyn, n_dyn);
}

static EVal load_lvalue(E *e, LVal lv) {
    EVal r = {0};
    MLIR_ValueHandle p = lval_address(e, lv);
    r.val = emit_load_v(e, p, lv.elem_ty);
    r.is_float = (lv.elem_ty == e->f32);
    r.is_i64 = (lv.elem_ty == e->i64);
    r.is_ptr = (lv.elem_ty == e->ptr);
    r.ptr_elem = MLIR_INVALID_HANDLE;
    return r;
}

static void store_lvalue(E *e, LVal lv, MLIR_ValueHandle v) {
    MLIR_ValueHandle p = lval_address(e, lv);
    emit_store_v(e, v, p);
}

// ----- struct-context walker: produces (base_ptr, source_elem, sd, path)
// for the struct lvalue being descended into (not yet a leaf).

typedef struct {
    bool             ok;
    MLIR_ValueHandle base_ptr;
    MLIR_TypeHandle  source_elem;
    StructDef       *sd;
    int32_t         *const_path;
    size_t           n_const_path;
    size_t           cap_const_path;
    MLIR_ValueHandle dyn_index;
} SCtx;

static void sctx_push(E *e, SCtx *c, int32_t v) {
    if (c->n_const_path == c->cap_const_path) {
        size_t cap = c->cap_const_path ? c->cap_const_path * 2 : 4;
        int32_t *np = arena_new_array(e->arena, int32_t, cap);
        for (size_t i = 0; i < c->n_const_path; i++) np[i] = c->const_path[i];
        c->const_path = np;
        c->cap_const_path = cap;
    }
    c->const_path[c->n_const_path++] = v;
}

static SCtx walk_struct_lhs(E *e, Scope *sc, Expr *ex) {
    e->cur_line = ex->line;
    SCtx r = {0};
    r.dyn_index = MLIR_INVALID_HANDLE;
    if (ex->kind == EX_VAR) {
        Sym *s = lookup(e, sc, ex->name);
        if (!s) { EMIT_ERR(e, "undefined variable: {}", ex->name); return r; }
        if (s->type.kind == TY_STRUCT) {
            r.base_ptr = sym_addr(e, s);
            r.source_elem = find_struct_type(e, s->sdef);
            r.sd = s->sdef;
            sctx_push(e, &r, 0);
            r.ok = (s->sdef != NULL);
            return r;
        }
        if (s->type.kind == TY_PTR_STRUCT) {
            // Load the inner ptr first.
            r.base_ptr = emit_load_v(e, sym_addr(e, s), e->ptr);
            r.source_elem = find_struct_type(e, s->sdef);
            r.sd = s->sdef;
            sctx_push(e, &r, 0);
            r.ok = (s->sdef != NULL);
            return r;
        }
        EMIT_ERR(e, "field/index access on non-struct variable {}", ex->name);
        return r;
    }
    if (ex->kind == EX_DEREF) {
        if (ex->lhs->kind == EX_VAR) {
            Sym *s = lookup(e, sc, ex->lhs->name);
            if (!s || s->type.kind != TY_PTR_STRUCT || !s->sdef) {
                EMIT_ERR(e, "-> requires a struct pointer");
                return r;
            }
            r.base_ptr = emit_load_v(e, sym_addr(e, s), e->ptr);
            r.source_elem = find_struct_type(e, s->sdef);
            r.sd = s->sdef;
            sctx_push(e, &r, 0);
            r.ok = true;
            return r;
        }
        // General case: evaluate the inner expression as a struct pointer.
        // Supports mid-chain walks like p->left->right and (*f(...)).x.
        EVal v = emit_expr(e, sc, ex->lhs);
        if (!v.is_ptr || !v.sdef) {
            EMIT_ERR(e, "-> requires a struct pointer");
            return r;
        }
        r.base_ptr = v.val;
        r.sd = v.sdef;
        r.source_elem = find_struct_type(e, v.sdef);
        sctx_push(e, &r, 0);
        r.ok = true;
        return r;
    }
    if (ex->kind == EX_INDEX && ex->lhs->kind == EX_VAR) {
        Sym *s = lookup(e, sc, ex->lhs->name);
        if (!s || s->type.kind != TY_ARRAY_STRUCT || !s->sdef) {
            EMIT_ERR(e, "arr[i].f requires an array of struct");
            return r;
        }
        MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
        r.base_ptr = sym_addr(e, s);
        // Source elem is the !llvm.array<N x !llvm.struct<...>>.
        MLIR_TypeHandle struct_ty = find_struct_type(e, s->sdef);
        r.source_elem = MLIR_CreateTypeLLVMArray(e->ctx, struct_ty, (uint64_t)s->type.array_len);
        r.sd = s->sdef;
        sctx_push(e, &r, 0);                // step into the alloca's one element
        sctx_push(e, &r, LLVM_GEP_DYN);     // dynamic array index
        r.dyn_index = idx_i32;
        r.ok = true;
        return r;
    }
    if (ex->kind == EX_FIELD) {
        SCtx parent = walk_struct_lhs(e, sc, ex->lhs);
        if (!parent.ok) return r;
        int idx = struct_field_index(parent.sd, ex->name);
        if (idx < 0) {
            EMIT_ERR(e, "unknown struct field {}", ex->name);
            return r;
        }
        Type ft = parent.sd->fields.data[idx].type;
        if (ft.kind != TY_STRUCT) {
            EMIT_ERR(e, "cannot chain field access through scalar field {}", ex->name);
            return r;
        }
        r = parent;
        sctx_push(e, &r, idx);
        r.sd = find_struct(e, ft.struct_name);
        r.ok = (r.sd != NULL);
        return r;
    }
    EMIT_ERR(e, "unsupported lvalue base in field access");
    return r;
}

static LVal emit_lvalue(E *e, Scope *sc, Expr *ex) {
    e->cur_line = ex->line;
    LVal r = {0};
    r.source_elem = MLIR_INVALID_HANDLE;
    r.dyn_index = MLIR_INVALID_HANDLE;
    r.elem_ty = e->i32;
    switch (ex->kind) {
        case EX_VAR: {
            Sym *s = lookup(e, sc, ex->name);
            if (!s) {
                EMIT_ERR(e, "undefined variable: {}", ex->name);
                return r;
            }
            r.base_ptr = sym_addr(e, s);
            if (s->type.kind == TY_F32) r.elem_ty = e->f32;
            else if (s->type.kind == TY_I64) r.elem_ty = e->i64;
            else if (s->type.kind == TY_PTR_STRUCT || s->type.kind == TY_PTR_I32 ||
                     s->type.kind == TY_PTR_CHAR || s->type.kind == TY_PTR_VOID ||
                     s->type.kind == TY_FNPTR || s->type.kind == TY_PTR_PTR)
                r.elem_ty = e->ptr;
            else r.elem_ty = e->i32;
            return r;
        }
        case EX_INDEX: {
            // Struct field array: s.f[i] / p->f[i] (1D) or s.f[i][j] (2D),
            // including chained walks like arr[k].f[i] (parent.dyn_index
            // is non-INVALID then). Emit a single GEP that descends from
            // the parent struct's source_elem all the way to the scalar
            // (or pointer) leaf.
            Expr *field_expr = NULL;
            Expr *idx_a = NULL, *idx_b = NULL;
            if (ex->lhs->kind == EX_FIELD) {
                field_expr = ex->lhs;
                idx_a = ex->rhs;
            } else if (ex->lhs->kind == EX_INDEX &&
                       ex->lhs->lhs->kind == EX_FIELD) {
                field_expr = ex->lhs->lhs;
                idx_a = ex->lhs->rhs;
                idx_b = ex->rhs;
            }
            if (field_expr) {
                SCtx parent = walk_struct_lhs(e, sc, field_expr->lhs);
                if (parent.ok) {
                    int fidx = struct_field_index(parent.sd, field_expr->name);
                    if (fidx < 0) {
                        EMIT_ERR(e, "unknown struct field {}",
                                field_expr->name);
                        return r;
                    }
                    Type ft = parent.sd->fields.data[fidx].type;
                    bool is_arr_i32 = (ft.kind == TY_ARRAY_I32);
                    bool is_arr_f32 = (ft.kind == TY_ARRAY_F32);
                    bool is_arr_pst = (ft.kind == TY_ARRAY_PTR_STRUCT);
                    bool is_arr_pch = (ft.kind == TY_ARRAY_PTR_CHAR);
                    if (is_arr_i32 || is_arr_f32 || is_arr_pst || is_arr_pch) {
                        bool is_2d = (ft.array_len2 != 0);
                        if (is_2d && !idx_b) {
                            EMIT_ERR(e, "2D field array requires both indices, e.g. s.m[i][j]");
                            return r;
                        }
                        if (!is_2d && idx_b) {
                            EMIT_ERR(e, "1D field array indexed twice");
                            return r;
                        }
                        if ((is_arr_pst || is_arr_pch) && is_2d) {
                            EMIT_ERR(e, "2D pointer-array fields are not supported");
                            return r;
                        }
                        MLIR_TypeHandle elem = is_arr_f32 ? e->f32
                                              : (is_arr_pst || is_arr_pch) ? e->ptr
                                              : e->i32;
                        MLIR_ValueHandle iv = emit_expr_i32(e, sc, idx_a);
                        MLIR_ValueHandle jv = is_2d ? emit_expr_i32(e, sc, idx_b)
                                                   : MLIR_INVALID_HANDLE;
                        size_t parent_n = parent.n_const_path;
                        size_t extra = 1 + (is_2d ? 2 : 1);
                        size_t total = parent_n + extra;
                        int32_t *path = arena_new_array(e->arena, int32_t, total);
                        for (size_t k = 0; k < parent_n; k++) path[k] = parent.const_path[k];
                        path[parent_n] = (int32_t)fidx;
                        path[parent_n + 1] = LLVM_GEP_DYN;
                        if (is_2d) path[parent_n + 2] = LLVM_GEP_DYN;
                        size_t n_dyn = (parent.dyn_index != MLIR_INVALID_HANDLE ? 1 : 0)
                                     + 1 + (is_2d ? 1 : 0);
                        MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, n_dyn);
                        size_t di = 0;
                        if (parent.dyn_index != MLIR_INVALID_HANDLE) dyn[di++] = parent.dyn_index;
                        dyn[di++] = iv;
                        if (is_2d) dyn[di++] = jv;
                        MLIR_ValueHandle gep = emit_gep(e, parent.base_ptr,
                            parent.source_elem, path, total, dyn, n_dyn);
                        r.base_ptr = gep;
                        r.elem_ty = elem;
                        return r;
                    }
                    // Pointer field indexed: `s.f[i]` / `p->f[i]` where
                    // `f` is `T *` (TY_PTR_I32 / TY_PTR_CHAR / TY_PTR_VOID
                    // / TY_PTR_PTR). Load the pointer from the field, then
                    // GEP it with the (single) index.
                    bool is_ptr_field = (ft.kind == TY_PTR_I32 ||
                                         ft.kind == TY_PTR_CHAR ||
                                         ft.kind == TY_PTR_VOID ||
                                         ft.kind == TY_PTR_PTR);
                    if (is_ptr_field && idx_a && !idx_b) {
                        size_t parent_n = parent.n_const_path;
                        size_t total = parent_n + 1;
                        int32_t *fpath = arena_new_array(e->arena, int32_t, total);
                        for (size_t k = 0; k < parent_n; k++) fpath[k] = parent.const_path[k];
                        fpath[parent_n] = (int32_t)fidx;
                        size_t n_dyn = (parent.dyn_index != MLIR_INVALID_HANDLE ? 1 : 0);
                        MLIR_ValueHandle *fdyn = arena_new_array(e->arena, MLIR_ValueHandle,
                                                                 n_dyn ? n_dyn : 1);
                        if (n_dyn) fdyn[0] = parent.dyn_index;
                        MLIR_ValueHandle field_addr = emit_gep(e, parent.base_ptr,
                            parent.source_elem, fpath, total, fdyn, n_dyn);
                        MLIR_ValueHandle base = emit_load_v(e, field_addr, e->ptr);
                        MLIR_TypeHandle elem = (ft.kind == TY_PTR_CHAR) ? e->i8
                                              : (ft.kind == TY_PTR_PTR) ? e->ptr
                                              : ft.ptr_is_i64 ? e->i64
                                              : e->i32;
                        MLIR_ValueHandle iv = emit_expr_i32(e, sc, idx_a);
                        int32_t *gpath = arena_new_array(e->arena, int32_t, 1);
                        gpath[0] = LLVM_GEP_DYN;
                        MLIR_ValueHandle *gdyn = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                        gdyn[0] = iv;
                        r.base_ptr = emit_gep(e, base, elem, gpath, 1, gdyn, 1);
                        r.elem_ty = elem;
                        return r;
                    }
                }
            }
            // Multi-dim array: m[i][j] for `int m[N1][N2];`.
            if (ex->lhs->kind == EX_INDEX &&
                ex->lhs->lhs->kind == EX_VAR) {
                Sym *s = lookup(e, sc, ex->lhs->lhs->name);
                if (s && s->type.kind == TY_ARRAY_I32 && s->type.array_len2 != 0) {
                    MLIR_ValueHandle i_v = emit_expr_i32(e, sc, ex->lhs->rhs);
                    MLIR_ValueHandle j_v = emit_expr_i32(e, sc, ex->rhs);
                    MLIR_TypeHandle inner_arr = MLIR_CreateTypeLLVMArray(
                        e->ctx, e->i32, (uint64_t)s->type.array_len2);
                    MLIR_TypeHandle outer_arr = MLIR_CreateTypeLLVMArray(
                        e->ctx, inner_arr, (uint64_t)s->type.array_len);
                    int32_t *path = arena_new_array(e->arena, int32_t, 3);
                    path[0] = 0; path[1] = LLVM_GEP_DYN; path[2] = LLVM_GEP_DYN;
                    MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, 2);
                    dyn[0] = i_v; dyn[1] = j_v;
                    MLIR_ValueHandle p = emit_gep(e, sym_addr(e, s), outer_arr,
                                                  path, 3, dyn, 2);
                    r.base_ptr = p;
                    r.elem_ty = e->i32;
                    return r;
                }
            }
            if (ex->lhs->kind != EX_VAR) {
                EMIT_ERR(e, "only simple-array indexing is supported");
                return r;
            }
            Sym *s = lookup(e, sc, ex->lhs->name);
            if (!s) {
                EMIT_ERR(e, "undefined array: {}", ex->lhs->name);
                return r;
            }
            if (s->type.kind == TY_ARRAY_STRUCT) {
                EMIT_ERR(e, "array-of-struct element must be field-accessed (arr[i].f)");
                return r;
            }
            // Pointer indexing: p[i] for int* / char* — GEP %p[%i].
            if (s->type.kind == TY_PTR_I32 || s->type.kind == TY_PTR_CHAR) {
                MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
                MLIR_ValueHandle base = emit_load_v(e, sym_addr(e, s), e->ptr);
                MLIR_TypeHandle elem = (s->type.kind == TY_PTR_CHAR) ? e->i8
                                      : s->type.ptr_is_i64 ? e->i64
                                      : e->i32;
                int32_t *path = arena_new_array(e->arena, int32_t, 1);
                path[0] = LLVM_GEP_DYN;
                MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                dyn[0] = idx_i32;
                r.base_ptr = emit_gep(e, base, elem, path, 1, dyn, 1);
                r.elem_ty = elem;
                return r;
            }
            if (s->type.kind != TY_ARRAY_I32) {
                EMIT_ERR(e, "indexing of non-array variable");
                return r;
            }
            if (s->type.array_len2 != 0) {
                EMIT_ERR(e, "2D array requires both indices, e.g. m[i][j]");
                return r;
            }
            MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
            r.base_ptr = sym_addr(e, s);
            r.source_elem = MLIR_CreateTypeLLVMArray(e->ctx, e->i32, (uint64_t)s->type.array_len);
            int32_t *path = arena_new_array(e->arena, int32_t, 2);
            path[0] = 0; path[1] = LLVM_GEP_DYN;
            r.const_path = path; r.n_const_path = 2;
            r.dyn_index = idx_i32;
            r.elem_ty = e->i32;
            return r;
        }
        case EX_DEREF: {
            if (ex->lhs->kind == EX_VAR) {
                Sym *s = lookup(e, sc, ex->lhs->name);
                if (!s || (s->type.kind != TY_PTR_I32 &&
                           s->type.kind != TY_PTR_CHAR &&
                           s->type.kind != TY_PTR_PTR)) {
                    EMIT_ERR(e, "dereference of non-pointer");
                    return r;
                }
                // Load the inner ptr from p's slot.
                r.base_ptr = emit_load_v(e, sym_addr(e, s), e->ptr);
                if (s->type.kind == TY_PTR_CHAR) r.elem_ty = e->i8;
                else if (s->type.kind == TY_PTR_PTR) r.elem_ty = e->ptr;
                else r.elem_ty = e->i32;
                return r;
            }
            // General `*<expr>` form (e.g. *(p+i)). Evaluate the operand
            // as a pointer expression; default elem type to i32.
            EVal v = emit_expr(e, sc, ex->lhs);
            if (!v.is_ptr) {
                EMIT_ERR(e, "dereference of non-pointer expression");
                return r;
            }
            r.base_ptr = v.val;
            r.elem_ty = (v.ptr_elem != MLIR_INVALID_HANDLE) ? v.ptr_elem : e->i32;
            return r;
        }
        case EX_FIELD: {
            SCtx parent = walk_struct_lhs(e, sc, ex->lhs);
            if (!parent.ok) return r;
            int idx = struct_field_index(parent.sd, ex->name);
            if (idx < 0) {
                EMIT_ERR(e, "unknown struct field {}", ex->name);
                return r;
            }
            Type ft = parent.sd->fields.data[idx].type;
            if (ft.kind != TY_I32 && ft.kind != TY_I64 && ft.kind != TY_F32 &&
                ft.kind != TY_PTR_STRUCT && ft.kind != TY_PTR_I32 &&
                ft.kind != TY_PTR_CHAR && ft.kind != TY_PTR_VOID &&
                ft.kind != TY_FNPTR && ft.kind != TY_PTR_PTR) {
                EMIT_ERR(e, "field {} is not a scalar lvalue", ex->name);
                return r;
            }
            sctx_push(e, &parent, idx);
            r.base_ptr = parent.base_ptr;
            r.source_elem = parent.source_elem;
            r.const_path = parent.const_path;
            r.n_const_path = parent.n_const_path;
            r.dyn_index = parent.dyn_index;
            if (ft.kind == TY_F32) r.elem_ty = e->f32;
            else if (ft.kind == TY_I64) r.elem_ty = e->i64;
            else if (ft.kind == TY_PTR_STRUCT || ft.kind == TY_PTR_I32 ||
                     ft.kind == TY_PTR_CHAR || ft.kind == TY_PTR_VOID ||
                     ft.kind == TY_FNPTR || ft.kind == TY_PTR_PTR) r.elem_ty = e->ptr;
            else r.elem_ty = e->i32;
            return r;
        }
        default:
            EMIT_ERR(e, "invalid lvalue");
            return r;
    }
}

static void unify_numeric(E *e, EVal *a, EVal *b) {
    if (a->is_float || b->is_float) {
        if (a->is_float == b->is_float) return;
        if (!a->is_float) {
            // Promote i64 -> i32 first if necessary, since emit_sitofp
            // assumes i32 input.
            if (a->is_i64) { a->val = emit_trunci_i64_to_i32(e, a->val); a->is_i64 = false; }
            a->val = emit_sitofp(e, a->val); a->is_float = true;
        }
        if (!b->is_float) {
            if (b->is_i64) { b->val = emit_trunci_i64_to_i32(e, b->val); b->is_i64 = false; }
            b->val = emit_sitofp(e, b->val); b->is_float = true;
        }
        return;
    }
    if (a->is_i64 == b->is_i64) return;
    if (!a->is_i64) { a->val = emit_extsi_i32_to_i64(e, a->val); a->is_i64 = true; }
    if (!b->is_i64) { b->val = emit_extsi_i32_to_i64(e, b->val); b->is_i64 = true; }
}

// ----- struct-by-pointer helpers -----

// Resolve a struct-typed call argument into the source !llvm.ptr.
// Accepts:
//   - EX_VAR struct          -> sym->addr
//   - EX_VAR struct*         -> load(sym->addr)
//   - EX_ADDR(EX_VAR struct) -> sym->addr
//   - any expression yielding a pointer (struct* field load, malloc cast,
//     null, etc.) -> the EVal's !llvm.ptr value
static MLIR_ValueHandle resolve_struct_source(E *e, Scope *sc, Expr *arg, StructDef **out_sd) {
    Expr *target = arg;
    if (arg->kind == EX_ADDR) {
        if (arg->lhs->kind != EX_VAR) {
            EMIT_ERR(e, "&expr in struct context requires a simple variable");
            return MLIR_INVALID_HANDLE;
        }
        target = arg->lhs;
    }
    if (target->kind == EX_VAR) {
        Sym *s = lookup(e, sc, target->name);
        if (s && (s->type.kind == TY_STRUCT || s->type.kind == TY_PTR_STRUCT) && s->sdef) {
            if (arg->kind == EX_ADDR && s->type.kind != TY_STRUCT) {
                EMIT_ERR(e, "&<var> requires a struct local");
                return MLIR_INVALID_HANDLE;
            }
            *out_sd = s->sdef;
            if (s->type.kind == TY_PTR_STRUCT) return emit_load_v(e, sym_addr(e, s), e->ptr);
            return sym_addr(e, s);
        }
    }
    // Fallback: evaluate as a generic pointer expression.
    EVal v = emit_expr(e, sc, arg);
    if (!v.is_ptr) {
        EMIT_ERR(e, "argument is not a struct or struct pointer");
        return MLIR_INVALID_HANDLE;
    }
    *out_sd = v.sdef;
    return v.val;
}

// Recursively copy each leaf of a struct from src to dst (both !llvm.ptr to
// the struct's storage), using the struct's identified type for GEP descent.
static void emit_struct_copy_path(E *e, MLIR_ValueHandle dst, MLIR_ValueHandle src,
                                  MLIR_TypeHandle source_elem, StructDef *sd,
                                  int32_t *prefix, size_t n_prefix) {
    for (size_t i = 0; i < sd->fields.size; i++) {
        Type ft = sd->fields.data[i].type;
        size_t n_path = n_prefix + 1;
        int32_t *path = arena_new_array(e->arena, int32_t, n_path);
        for (size_t k = 0; k < n_prefix; k++) path[k] = prefix[k];
        path[n_prefix] = (int32_t)i;
        if (ft.kind == TY_STRUCT) {
            StructDef *inner = find_struct(e, ft.struct_name);
            emit_struct_copy_path(e, dst, src, source_elem, inner, path, n_path);
        } else if (ft.kind == TY_PTR_STRUCT || ft.kind == TY_PTR_I32 ||
                   ft.kind == TY_PTR_CHAR || ft.kind == TY_PTR_VOID) {
            MLIR_ValueHandle sp = emit_gep(e, src, source_elem, path, n_path, NULL, 0);
            MLIR_ValueHandle val = emit_load_v(e, sp, e->ptr);
            MLIR_ValueHandle dp = emit_gep(e, dst, source_elem, path, n_path, NULL, 0);
            emit_store_v(e, val, dp);
        } else if (ft.kind == TY_ARRAY_I32 || ft.kind == TY_ARRAY_F32 ||
                   ft.kind == TY_ARRAY_PTR_STRUCT || ft.kind == TY_ARRAY_PTR_CHAR) {
            bool is_2d = (ft.array_len2 != 0);
            int64_t n1 = ft.array_len;
            int64_t n2 = is_2d ? ft.array_len2 : 1;
            MLIR_TypeHandle elem;
            if (ft.kind == TY_ARRAY_F32) elem = e->f32;
            else if (ft.kind == TY_ARRAY_PTR_STRUCT || ft.kind == TY_ARRAY_PTR_CHAR) elem = e->ptr;
            else elem = e->i32;
            for (int64_t a = 0; a < n1; a++) {
                for (int64_t b = 0; b < n2; b++) {
                    size_t inner_n = is_2d ? 2 : 1;
                    size_t total = n_path + inner_n;
                    int32_t *p2 = arena_new_array(e->arena, int32_t, total);
                    for (size_t kk = 0; kk < n_path; kk++) p2[kk] = path[kk];
                    p2[n_path] = (int32_t)a;
                    if (is_2d) p2[n_path + 1] = (int32_t)b;
                    MLIR_ValueHandle sp = emit_gep(e, src, source_elem, p2, total, NULL, 0);
                    MLIR_ValueHandle val = emit_load_v(e, sp, elem);
                    MLIR_ValueHandle dp = emit_gep(e, dst, source_elem, p2, total, NULL, 0);
                    emit_store_v(e, val, dp);
                }
            }
        } else {
            MLIR_TypeHandle elem = scalar_mlir_type(e, ft.kind);
            MLIR_ValueHandle sp = emit_gep(e, src, source_elem, path, n_path, NULL, 0);
            MLIR_ValueHandle val = emit_load_v(e, sp, elem);
            MLIR_ValueHandle dp = emit_gep(e, dst, source_elem, path, n_path, NULL, 0);
            emit_store_v(e, val, dp);
        }
    }
}

static void emit_struct_copy(E *e, MLIR_ValueHandle dst, MLIR_ValueHandle src, StructDef *sd) {
    MLIR_TypeHandle st = find_struct_type(e, sd);
    int32_t *prefix = arena_new_array(e->arena, int32_t, 1);
    prefix[0] = 0;
    emit_struct_copy_path(e, dst, src, st, sd, prefix, 1);
}

// Emit a func.call (non-variadic) or llvm.call (variadic). For
// struct-returning functions a hidden first sret !llvm.ptr operand is
// prepended; the function returns void; out_results[0] (if non-NULL) is
// set to that sret pointer. If out_sret_buf is non-INVALID, that ptr is
// used as the sret buffer instead of allocating a fresh one.
//
// For variadic callees the operand list may have MORE entries than
// sig->n_params; trailing args are appended verbatim (with the standard C
// promotion of f32 to f64 — currently rejected with a clear error since
// tinyC has no double type).
static void emit_flat_call(E *e, Scope *sc, FuncSig *sig, VecExprPtr args,
                           MLIR_ValueHandle *out_results,
                           MLIR_ValueHandle out_sret_buf) {
    if (!sig->is_variadic && args.size != sig->n_params) {
        EMIT_ERR(e, "call to {} arity mismatch", sig->name);
    }
    if (sig->is_variadic && args.size < sig->n_params) {
        EMIT_ERR(e, "call to variadic {} requires at least {} fixed arg(s)",
                 sig->name, (int64_t)sig->n_params);
    }
    size_t n_fixed = sig->n_flat_in;
    size_t n_extra = sig->is_variadic ? (args.size - sig->n_params) : 0;
    size_t n_in = n_fixed + n_extra;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, n_in ? n_in : 1);
    size_t op_off = 0;
    MLIR_ValueHandle ret_buf = MLIR_INVALID_HANDLE;
    if (sig->sret) {
        if (out_sret_buf != MLIR_INVALID_HANDLE) {
            ret_buf = out_sret_buf;
        } else {
            MLIR_TypeHandle st = find_struct_type(e, sig->ret.sdef);
            ret_buf = emit_alloca(e, st);
        }
        ops[op_off++] = ret_buf;
    }
    for (size_t i = 0; i < sig->n_params && i < args.size; i++) {
        SlotInfo *p = &sig->params[i];
        if (p->type.kind == TY_STRUCT) {
            // Caller-byval: allocate a fresh copy of the struct, copy from
            // the source, pass !llvm.ptr to the copy.
            StructDef *sd = NULL;
            MLIR_ValueHandle src = resolve_struct_source(e, sc, args.data[i], &sd);
            if (src == MLIR_INVALID_HANDLE || sd != p->sdef) {
                EMIT_ERR(e, "struct call arg type mismatch");
                ops[op_off++] = src;
                continue;
            }
            MLIR_TypeHandle st = find_struct_type(e, sd);
            MLIR_ValueHandle tmp = emit_alloca(e, st);
            emit_struct_copy(e, tmp, src, sd);
            ops[op_off++] = tmp;
        } else if (p->type.kind == TY_PTR_STRUCT) {
            StructDef *sd = NULL;
            MLIR_ValueHandle src = resolve_struct_source(e, sc, args.data[i], &sd);
            ops[op_off++] = src;
        } else {
            EVal v = emit_expr(e, sc, args.data[i]);
            ops[op_off++] = coerce_eval(e, v, scalar_mlir_type(e, p->type.kind));
        }
    }
    // Variadic-portion arguments: pass-through with no implicit conversion.
    // f32 is intentionally rejected (C requires float→double promotion and
    // tinyC has no double type yet).
    for (size_t i = sig->n_params; i < args.size; i++) {
        EVal v = emit_expr(e, sc, args.data[i]);
        if (v.is_float) {
            EMIT_ERR(e, "passing float through ... is not supported "
                        "(no double type in tinyC)");
        }
        ops[op_off++] = v.val;
    }
    size_t n_out = sig->n_flat_out;
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, n_out ? n_out : 1);
    MLIR_ValueHandle *rs  = arena_new_array(e->arena, MLIR_ValueHandle, n_out ? n_out : 1);
    for (size_t i = 0; i < n_out; i++) {
        rts[i] = sig->flat_out_tys[i];
        rs[i]  = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, (uint32_t)i,
                                          rts[i], ssa_name(e), eloc(e, 0));
    }
    MLIR_AttributeHandle callee_attr = MLIR_CreateAttributeSymbolRef(
        e->ctx, str_lit("callee"), sig->name);
    if (sig->is_variadic) {
        MLIR_AttributeHandle var_ty = MLIR_CreateAttributeType(
            e->ctx, str_lit("var_callee_type"), sig->llvm_fn_ty);
        MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 2);
        as[0] = callee_attr; as[1] = var_ty;
        emit_op(e, OP_TYPE_UNREGISTERED, str_lit("llvm.call"),
                rts, n_out, rs, n_out, ops, op_off, as, 2, NULL, 0);
    } else {
        MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1);
        as[0] = callee_attr;
        emit_op(e, OP_TYPE_FUNC_CALL, str_lit("func.call"),
                rts, n_out, rs, n_out, ops, op_off, as, 1, NULL, 0);
    }
    if (out_results) {
        if (sig->sret) out_results[0] = ret_buf;
        else for (size_t i = 0; i < n_out; i++) out_results[i] = rs[i];
    }
}

// Compile-time folder for literal-only integer expressions. Returns
// true and writes the folded i32 value (sign-extended in int64_t) when
// `ex` and all its sub-expressions are pure integer-only computations
// reducible at emit time. Used to produce a single `arith.constant`
// instead of a chain of arith ops for things like `1 + 2 * 3`,
// `(1 << 4) | 3`, `sizeof(int) * 4`, or ternary with a literal cond.
//
// Folded operations (i32 semantics, wrap-around on overflow):
//   EX_INT, EX_SIZEOF
//   EX_UN: OP_NEG, OP_BNOT, OP_NOT
//   EX_BIN: + - * / % & | ^ << >> < <= > >= == != && ||
//   EX_TERNARY when the condition itself folds.
//
// Refused (returns false; caller falls back to normal emission):
//   - EX_VAR, EX_CALL, EX_ASSIGN, EX_INDEX, EX_ADDR, EX_DEREF,
//     EX_FIELD, EX_FLOAT, EX_NULL, EX_STR, EX_CAST
//   - division / modulo by zero (let runtime / LLVM handle it)
//   - shifts by negative or >=32 (let LLVM handle it)
static bool ast_fold_int(E *e, Expr *ex, int64_t *out) {
    if (!ex) return false;
    switch (ex->kind) {
        case EX_INT:
            *out = (int64_t)(int32_t)ex->int_value;
            return true;
        case EX_SIZEOF:
            if (ex->cast_type.kind == TY_VOID) return false;
            *out = (int64_t)(int32_t)type_size(e, ex->cast_type);
            return true;
        case EX_UN: {
            int64_t a;
            if (!ast_fold_int(e, ex->lhs, &a)) return false;
            int32_t a32 = (int32_t)a;
            switch (ex->op) {
                case OP_NEG:  *out = (int64_t)(int32_t)(-a32);   return true;
                case OP_BNOT: *out = (int64_t)(int32_t)(~a32);   return true;
                case OP_NOT:  *out = (a32 == 0) ? 1 : 0;         return true;
                default: return false;
            }
        }
        case EX_BIN: {
            int64_t a, b;
            if (!ast_fold_int(e, ex->lhs, &a)) return false;
            if (!ast_fold_int(e, ex->rhs, &b)) return false;
            int32_t a32 = (int32_t)a, b32 = (int32_t)b;
            switch (ex->op) {
                case OP_ADD: *out = (int64_t)(int32_t)(a32 + b32); return true;
                case OP_SUB: *out = (int64_t)(int32_t)(a32 - b32); return true;
                case OP_MUL: *out = (int64_t)(int32_t)(a32 * b32); return true;
                case OP_DIV:
                    if (b32 == 0) return false;
                    *out = (int64_t)(int32_t)(a32 / b32); return true;
                case OP_MOD:
                    if (b32 == 0) return false;
                    *out = (int64_t)(int32_t)(a32 % b32); return true;
                case OP_BAND: *out = (int64_t)(int32_t)(a32 & b32); return true;
                case OP_BOR:  *out = (int64_t)(int32_t)(a32 | b32); return true;
                case OP_BXOR: *out = (int64_t)(int32_t)(a32 ^ b32); return true;
                case OP_SHL:
                    if (b32 < 0 || b32 >= 32) return false;
                    *out = (int64_t)(int32_t)((uint32_t)a32 << b32); return true;
                case OP_SHR:
                    if (b32 < 0 || b32 >= 32) return false;
                    *out = (int64_t)(int32_t)(a32 >> b32); return true;
                case OP_LT: *out = a32 <  b32 ? 1 : 0; return true;
                case OP_LE: *out = a32 <= b32 ? 1 : 0; return true;
                case OP_GT: *out = a32 >  b32 ? 1 : 0; return true;
                case OP_GE: *out = a32 >= b32 ? 1 : 0; return true;
                case OP_EQ: *out = a32 == b32 ? 1 : 0; return true;
                case OP_NE: *out = a32 != b32 ? 1 : 0; return true;
                case OP_AND: *out = (a32 != 0 && b32 != 0) ? 1 : 0; return true;
                case OP_OR:  *out = (a32 != 0 || b32 != 0) ? 1 : 0; return true;
                default: return false;
            }
        }
        case EX_TERNARY: {
            int64_t c;
            if (!ast_fold_int(e, ex->lhs, &c)) return false;
            Expr *pick = ((int32_t)c != 0) ? ex->rhs : ex->lvalue;
            return ast_fold_int(e, pick, out);
        }
        default:
            return false;
    }
}

// True if the i-th `_Generic` association's type matches the controlling
// expression type. struct types match by struct_name; integer types
// match by kind plus, when both have explicit int_bits, equal width and
// sign. Pointer kinds match by exact kind.
static bool generic_type_matches(Type a, Type b) {
    if (a.kind != b.kind) return false;
    if (a.kind == TY_STRUCT || a.kind == TY_PTR_STRUCT) {
        return str_eq(a.struct_name, b.struct_name);
    }
    if (a.kind == TY_I32 || a.kind == TY_I64) {
        if (a.int_bits != 0 && b.int_bits != 0 && a.int_bits != b.int_bits)
            return false;
        return true;
    }
    return true;
}

static Expr *generic_select(E *e, Scope *sc, Expr *ex) {
    Type ct = infer_expr_type(e, sc, ex->lhs);
    int n = (int)ex->args.size;
    int chosen = -1;
    for (int i = 0; i < n; i++) {
        if (i == ex->generic_default_index) continue;
        if (generic_type_matches(ex->generic_types[i], ct)) {
            chosen = i; break;
        }
    }
    if (chosen < 0) chosen = ex->generic_default_index;
    if (chosen < 0 || chosen >= n) {
        EMIT_ERR(e, "_Generic: no matching association and no default");
        return ex->lhs;
    }
    return ex->args.data[chosen];
}

static EVal emit_expr(E *e, Scope *sc, Expr *ex) {
    e->cur_line = ex->line;
    EVal r = {0};
    if (ex->kind == EX_GENERIC) {
        return emit_expr(e, sc, generic_select(e, sc, ex));
    }
    // Pre-pass: literal-only integer subtree -> a single arith.constant.
    // Only kicks in for nodes that ast_fold_int actually folds; anything
    // with a side effect or a non-int operand falls through unchanged.
    if (ex->kind == EX_BIN || ex->kind == EX_UN || ex->kind == EX_TERNARY) {
        int64_t fv;
        if (ast_fold_int(e, ex, &fv)) {
            r.val = emit_const_i32(e, fv);
            return r;
        }
    }
    switch (ex->kind) {
        case EX_INT:
            if (ex->is_i64) {
                r.val = emit_const_i64(e, ex->int_value);
                r.is_i64 = true;
            } else {
                r.val = emit_const_i32(e, ex->int_value);
            }
            return r;
        case EX_FLOAT:
            r.val = emit_const_f32(e, ex->float_value); r.is_float = true; return r;
        case EX_NULL:
            r.val = emit_null_ptr(e); r.is_ptr = true; r.sdef = NULL; return r;
        case EX_VA_ARG: {
            // MLIR's LLVM dialect does not expose a `va_arg` op (only the
            // start/end/copy intrinsics) — and clang traditionally lowers
            // va_arg manually because its IR variant is buggy on x86_64.
            // We side-step both problems by calling small runtime helpers
            // (defined in runtime.c) that wrap stdarg.h; the helpers
            // receive a !llvm.ptr to our 32-byte va_list buffer.
            if (!ex->lhs || ex->lhs->kind != EX_VAR) {
                EMIT_ERR(e, "va_arg first argument must be a va_list variable");
                r.val = emit_const_i32(e, 0); return r;
            }
            Sym *sy = lookup(e, sc, ex->lhs->name);
            if (!sy || sy->type.kind != TY_VA_LIST) {
                EMIT_ERR(e, "va_arg first argument must be of type va_list");
                r.val = emit_const_i32(e, 0); return r;
            }
            TypeKind tk = ex->cast_type.kind;
            string helper;
            MLIR_TypeHandle rt;
            switch (tk) {
                case TY_I32:        helper = str_lit("tinyc_va_arg_i32"); rt = e->i32; break;
                case TY_I64:        helper = str_lit("tinyc_va_arg_i64"); rt = e->i64; break;
                case TY_PTR_CHAR:
                case TY_PTR_VOID:
                case TY_PTR_I32:
                case TY_PTR_STRUCT:
                case TY_PTR_PTR:    helper = str_lit("tinyc_va_arg_ptr"); rt = e->ptr; break;
                default:
                    EMIT_ERR(e, "va_arg only supports int, long, and pointer types");
                    r.val = emit_const_i32(e, 0); return r;
            }
            MLIR_ValueHandle ap_ptr = sym_addr(e, sy);
            MLIR_ValueHandle res = MLIR_CreateValueOpResult(
                e->ctx, MLIR_INVALID_HANDLE, 0, rt, ssa_name(e), eloc(e, 0));
            MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = rt;
            MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = res;
            MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = ap_ptr;
            MLIR_AttributeHandle ca = MLIR_CreateAttributeSymbolRef(
                e->ctx, str_lit("callee"), helper);
            MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1);
            as[0] = ca;
            emit_op(e, OP_TYPE_FUNC_CALL, str_lit("func.call"),
                    rts, 1, rs, 1, ops, 1, as, 1, NULL, 0);
            // Mark the helper as needed so that we emit its extern decl at
            // module scope.
            e->need_va_arg_helpers = true;
            r.val = res;
            r.is_i64 = (tk == TY_I64);
            r.is_ptr = (tk == TY_PTR_CHAR || tk == TY_PTR_VOID || tk == TY_PTR_I32
                        || tk == TY_PTR_STRUCT || tk == TY_PTR_PTR);
            r.is_str = (tk == TY_PTR_CHAR);
            r.is_void_ptr = (tk == TY_PTR_VOID);
            return r;
        }
        case EX_STR: {
            // String literal: emit a !llvm.ptr to a deduplicated
            // module-level llvm.mlir.global string buffer.
            string sym = intern_string(e, ex->name);
            r.val = emit_addressof(e, sym);
            r.is_ptr = true;
            r.is_str = true;
            return r;
        }
        case EX_TERNARY: {
            // Lower `c ? a : b` via cf.cond_br + alloca for result.
            EVal cv = emit_expr(e, sc, ex->lhs);
            MLIR_ValueHandle cb = emit_to_bool_i1(e, cv);
            MLIR_BlockHandle then_blk = new_cfg_block(e);
            MLIR_BlockHandle else_blk = new_cfg_block(e);
            MLIR_BlockHandle merge_blk = new_cfg_block(e);

            MLIR_BlockHandle save = e->cur_block;
            e->cur_block = then_blk; e->terminated = false;
            EVal av = emit_expr(e, sc, ex->rhs);
            MLIR_BlockHandle then_tail = e->cur_block;

            e->cur_block = else_blk; e->terminated = false;
            EVal bv = emit_expr(e, sc, ex->lvalue);
            MLIR_BlockHandle else_tail = e->cur_block;

            bool any_float = av.is_float || bv.is_float;
            bool any_ptr   = av.is_ptr   || bv.is_ptr;
            MLIR_TypeHandle rty = any_ptr ? e->ptr : (any_float ? e->f32 : e->i32);

            // Alloca in current (pre-branch) block.
            e->cur_block = save; e->terminated = false;
            MLIR_ValueHandle slot = emit_alloca(e, rty);

            // Coerce + store in then-tail.
            e->cur_block = then_tail; e->terminated = false;
            MLIR_ValueHandle ta = any_ptr
                ? (av.is_ptr ? av.val : emit_null_ptr(e))
                : coerce_eval(e, av, rty);
            emit_store_v(e, ta, slot);
            emit_branch(e, merge_blk);

            e->cur_block = else_tail; e->terminated = false;
            MLIR_ValueHandle tb = any_ptr
                ? (bv.is_ptr ? bv.val : emit_null_ptr(e))
                : coerce_eval(e, bv, rty);
            emit_store_v(e, tb, slot);
            emit_branch(e, merge_blk);

            // From the original block, branch on cb to then/else.
            e->cur_block = save; e->terminated = false;
            emit_cond_branch(e, cb, then_blk, else_blk);

            e->cur_block = merge_blk; e->terminated = false;
            r.val = emit_load_v(e, slot, rty);
            r.is_float = any_float;
            r.is_ptr   = any_ptr;
            r.is_str   = av.is_str && bv.is_str;
            r.sdef     = av.sdef ? av.sdef : bv.sdef;
            return r;
        }
        case EX_SIZEOF: {
            Type ty;
            if (ex->sizeof_is_expr) {
                ty = infer_expr_type(e, sc, ex->lhs);
                if (ty.kind == TY_VOID) {
                    EMIT_ERR(e, "sizeof of unsupported expression");
                    r.val = emit_const_i32(e, 1); return r;
                }
            } else {
                ty = ex->cast_type;
                if (ty.kind == TY_VOID) {
                    EMIT_ERR(e, "sizeof(void) is not allowed");
                    r.val = emit_const_i32(e, 1); return r;
                }
            }
            r.val = emit_const_i32(e, type_size(e, ty)); return r;
        }
        case EX_CAST: {
            EVal v = emit_expr(e, sc, ex->lhs);
            TypeKind ck = ex->cast_type.kind;
            bool ck_is_ptr = (ck == TY_PTR_VOID || ck == TY_PTR_I32 ||
                              ck == TY_PTR_CHAR || ck == TY_PTR_STRUCT ||
                              ck == TY_FNPTR || ck == TY_PTR_PTR);
            // Integer-to-integer cast: convert between i32 and i64; everything
            // else is treated as i32 (TY_I8/TY_I16 etc. fold to i32 here).
            if (!ck_is_ptr && !v.is_ptr) {
                bool want_i64 = (ck == TY_I64);
                if (want_i64 && !v.is_i64) {
                    v.val = emit_extsi_i32_to_i64(e, v.val);
                    v.is_i64 = true;
                } else if (!want_i64 && v.is_i64) {
                    v.val = emit_trunci_i64_to_i32(e, v.val);
                    v.is_i64 = false;
                }
                v.is_float = (ck == TY_F32) ? v.is_float : false;
                return v;
            }
            // Pointer-to-pointer cast: opaque !llvm.ptr is universal, so
            // just evaluate the operand. We tag the result type from
            // cast_type for downstream consumers.
            if (!v.is_ptr) {
                // Integer-to-pointer cast: only literal 0 is allowed
                // (used for `(T*)0` null sentinels in C). Synthesize a
                // null pointer.
                if (ex->lhs->kind == EX_INT && ex->lhs->int_value == 0) {
                    v.val = emit_null_ptr(e);
                } else {
                    EMIT_ERR(e, "cast operand is not a pointer");
                }
            }
            v.is_ptr = true;
            v.is_float = false;
            v.is_void_ptr = (ck == TY_PTR_VOID);
            if (ck == TY_PTR_STRUCT) {
                v.sdef = find_struct(e, ex->cast_type.struct_name);
            } else {
                v.sdef = NULL;
            }
            // Reset element-type hints for non-void typed pointers.
            if (ck == TY_PTR_I32) { v.ptr_elem = e->i32; v.is_str = false; }
            else if (ck == TY_PTR_CHAR) { v.ptr_elem = e->i8; v.is_str = true; }
            else if (ck == TY_PTR_VOID) { v.ptr_elem = MLIR_INVALID_HANDLE; v.is_str = false; }
            return v;
        }
        case EX_VAR:
        case EX_INDEX:
        case EX_DEREF:
        case EX_FIELD: {
            // Bare function name (no call): decay to a function-pointer
            // value via `llvm.mlir.addressof @sym : !llvm.ptr`. Tag the
            // result with a TY_FNPTR carrying the function's signature so
            // downstream call sites can build the right call_indirect.
            if (ex->kind == EX_VAR) {
                Sym *s = lookup(e, sc, ex->name);
                if (!s) {
                    FuncSig *fsig = find_sig(e, ex->name);
                    if (fsig) {
                        // Build the function MLIR type for func.constant.
                        size_t np = fsig->n_params;
                        MLIR_TypeHandle *in_tys = arena_new_array(e->arena,
                            MLIR_TypeHandle, np ? np : 1);
                        for (size_t i = 0; i < np; i++)
                            in_tys[i] = scalar_mlir_type(e, fsig->params[i].type.kind);
                        MLIR_TypeHandle ret_ty = scalar_mlir_type(e,
                            fsig->ret.type.kind);
                        MLIR_TypeHandle out_tys[1] = { ret_ty };
                        MLIR_TypeHandle f_ty = MLIR_CreateTypeFunction(e->ctx,
                            in_tys, np, out_tys, 1);
                        // %fn = func.constant @sym : (T...) -> R
                        MLIR_ValueHandle fn_v = MLIR_CreateValueOpResult(
                            e->ctx, MLIR_INVALID_HANDLE, 0, f_ty, ssa_name(e),
                            eloc(e, 0));
                        MLIR_TypeHandle *frts = arena_new_array(e->arena,
                            MLIR_TypeHandle, 1); frts[0] = f_ty;
                        MLIR_ValueHandle *frs = arena_new_array(e->arena,
                            MLIR_ValueHandle, 1); frs[0] = fn_v;
                        MLIR_AttributeHandle va = MLIR_CreateAttributeSymbolRef(
                            e->ctx, str_lit("value"), fsig->name);
                        MLIR_AttributeHandle *fas = arena_new_array(e->arena,
                            MLIR_AttributeHandle, 1); fas[0] = va;
                        emit_op(e, OP_TYPE_FUNC_CONSTANT, str_lit("func.constant"),
                                frts, 1, frs, 1, NULL, 0, fas, 1, NULL, 0);
                        // Cast (T...) -> R into !llvm.ptr for uniform storage.
                        MLIR_ValueHandle ptr_v = MLIR_CreateValueOpResult(
                            e->ctx, MLIR_INVALID_HANDLE, 0, e->ptr, ssa_name(e),
                            eloc(e, 0));
                        MLIR_TypeHandle *crts = arena_new_array(e->arena,
                            MLIR_TypeHandle, 1); crts[0] = e->ptr;
                        MLIR_ValueHandle *crs = arena_new_array(e->arena,
                            MLIR_ValueHandle, 1); crs[0] = ptr_v;
                        MLIR_ValueHandle *cops = arena_new_array(e->arena,
                            MLIR_ValueHandle, 1); cops[0] = fn_v;
                        emit_op(e, OP_TYPE_UNREALIZED_CONVERSION_CAST,
                                str_lit("builtin.unrealized_conversion_cast"),
                                crts, 1, crs, 1, cops, 1, NULL, 0, NULL, 0);
                        r.val = ptr_v;
                        r.is_ptr = true;
                        Type *fnty = arena_new(e->arena, Type);
                        *fnty = (Type){0};
                        fnty->kind = TY_FNPTR;
                        Type *ret_box = arena_new(e->arena, Type);
                        *ret_box = fsig->ret.type;
                        fnty->fnptr_ret = ret_box;
                        fnty->fnptr_nparams = (int)fsig->n_params;
                        fnty->fnptr_params = arena_new_array(e->arena, Type,
                            fnty->fnptr_nparams ? (size_t)fnty->fnptr_nparams : 1);
                        for (size_t i = 0; i < fsig->n_params; i++)
                            fnty->fnptr_params[i] = fsig->params[i].type;
                        r.fnptr_ty = fnty;
                        return r;
                    }
                    // Fall back to enum constants ONLY when both local
                    // scope and the function table miss — this preserves
                    // shadowing by inner-scope variables.
                    int64_t en_v;
                    if (find_enum(e, ex->name, &en_v)) {
                        r.val = emit_const_i32(e, en_v);
                        return r;
                    }
                }
            }
            LVal lv = emit_lvalue(e, sc, ex);
            EVal v = load_lvalue(e, lv);
            // Tag the resulting pointer with its target StructDef, when
            // determinable, so callers (struct-arg passing, pointer
            // comparisons) can reason about it.
            if (v.is_ptr) {
                if (ex->kind == EX_VAR) {
                    Sym *s = lookup(e, sc, ex->name);
                    if (s) {
                        v.sdef = s->sdef;
                        if (s->type.kind == TY_PTR_CHAR) { v.is_str = true; v.ptr_elem = e->i8; }
                        else if (s->type.kind == TY_PTR_I32) v.ptr_elem = e->i32;
                        else if (s->type.kind == TY_PTR_VOID) v.is_void_ptr = true;
                        else if (s->type.kind == TY_PTR_PTR) {
                            // Reading the value of a T** yields a T* — tag
                            // the EVal so a subsequent dereference can load
                            // the inner T at the right element type. The
                            // pointee field carries the inner pointer's
                            // element kind.
                            v.ptr_elem = e->ptr;
                            if (s->type.pointee) {
                                Type *pe = s->type.pointee;
                                if (pe->kind == TY_PTR_CHAR) v.is_str = true;
                                else if (pe->kind == TY_PTR_STRUCT) {
                                    v.sdef = find_struct(e, pe->struct_name);
                                } else if (pe->kind == TY_FNPTR) {
                                    Type *fnty = arena_new(e->arena, Type);
                                    *fnty = *pe;
                                    v.fnptr_ty = fnty;
                                }
                            }
                        }
                        else if (s->type.kind == TY_FNPTR) {
                            Type *fnty = arena_new(e->arena, Type);
                            *fnty = s->type;
                            v.fnptr_ty = fnty;
                        }
                    }
                } else if (ex->kind == EX_DEREF && ex->lhs->kind == EX_VAR) {
                    // *pp where pp is T**: the loaded value is a T*. Tag
                    // it from the pointee so further reads / calls work.
                    Sym *s = lookup(e, sc, ex->lhs->name);
                    if (s && s->type.kind == TY_PTR_PTR && s->type.pointee) {
                        Type *pe = s->type.pointee;
                        if (pe->kind == TY_PTR_I32) v.ptr_elem = e->i32;
                        else if (pe->kind == TY_PTR_CHAR) { v.is_str = true; v.ptr_elem = e->i8; }
                        else if (pe->kind == TY_PTR_VOID) v.is_void_ptr = true;
                        else if (pe->kind == TY_PTR_STRUCT) {
                            v.sdef = find_struct(e, pe->struct_name);
                        } else if (pe->kind == TY_FNPTR) {
                            Type *fnty = arena_new(e->arena, Type);
                            *fnty = *pe;
                            v.fnptr_ty = fnty;
                        }
                    }
                } else if (ex->kind == EX_FIELD) {
                    SCtx parent = walk_struct_lhs(e, sc, ex->lhs);
                    if (parent.ok) {
                        int fidx = struct_field_index(parent.sd, ex->name);
                        if (fidx >= 0) {
                            Type ft = parent.sd->fields.data[fidx].type;
                            if (ft.kind == TY_PTR_STRUCT) {
                                v.sdef = find_struct(e, ft.struct_name);
                            }
                        }
                    }
                } else if (ex->kind == EX_INDEX) {
                    // Reading an element of a struct-field array of
                    // pointers: tag the result so downstream consumers
                    // (print(), -> chains) see the right element kind.
                    Expr *fe = NULL;
                    if (ex->lhs->kind == EX_FIELD) fe = ex->lhs;
                    else if (ex->lhs->kind == EX_INDEX &&
                             ex->lhs->lhs->kind == EX_FIELD) fe = ex->lhs->lhs;
                    if (fe) {
                        SCtx parent = walk_struct_lhs(e, sc, fe->lhs);
                        if (parent.ok) {
                            int fidx = struct_field_index(parent.sd, fe->name);
                            if (fidx >= 0) {
                                Type ft = parent.sd->fields.data[fidx].type;
                                if (ft.kind == TY_ARRAY_PTR_CHAR) {
                                    v.is_str = true;
                                    v.ptr_elem = e->i8;
                                } else if (ft.kind == TY_ARRAY_PTR_STRUCT) {
                                    v.sdef = find_struct(e, ft.struct_name);
                                }
                            }
                        }
                    }
                }
            }
            return v;
        }
        case EX_ADDR: {
            // &x: result is the !llvm.ptr to x's storage. Used as the
            // RHS of `int* p = &x;` style decls and for &a[0]; treated as
            // a pseudo-rvalue whose `val` is a !llvm.ptr SSA value.
            if (ex->lhs->kind == EX_VAR) {
                Sym *s = lookup(e, sc, ex->lhs->name);
                if (!s) {
                    // &funcname: explicit address-of-function. Same as
                    // bare function-name decay.
                    FuncSig *fsig = find_sig(e, ex->lhs->name);
                    if (fsig) {
                        return emit_expr(e, sc, ex->lhs);
                    }
                    EMIT_ERR(e, "undefined variable: {}", ex->lhs->name);
                    return r;
                }
                r.val = sym_addr(e, s);
                r.is_ptr = true;
                if (s->type.kind == TY_I32 || s->type.kind == TY_ARRAY_I32) r.ptr_elem = e->i32;
                else if (s->type.kind == TY_F32) r.ptr_elem = e->f32;
                else if (s->type.kind == TY_PTR_I32 || s->type.kind == TY_PTR_CHAR ||
                         s->type.kind == TY_PTR_VOID || s->type.kind == TY_PTR_STRUCT ||
                         s->type.kind == TY_FNPTR || s->type.kind == TY_PTR_PTR) {
                    // &p where p is itself a pointer: yield a T** (storage
                    // !llvm.ptr) whose deref loads another !llvm.ptr.
                    r.ptr_elem = e->ptr;
                }
                return r;
            }
            if (ex->lhs->kind == EX_INDEX) {
                // &arr[i] -> GEP address.
                LVal lv = emit_lvalue(e, sc, ex->lhs);
                r.val = lval_address(e, lv);
                r.is_ptr = true;
                r.ptr_elem = lv.elem_ty;
                return r;
            }
            EMIT_ERR(e, "unsupported & operand");
            return r;
        }
        case EX_ASSIGN: {
            // Struct rhs from a struct-returning call: q = f(args);
            if (ex->lvalue->kind == EX_VAR && ex->rhs_assign->kind == EX_CALL) {
                Sym *target = lookup(e, sc, ex->lvalue->name);
                FuncSig *sig = find_sig(e, ex->rhs_assign->callee);
                if (target && target->type.kind == TY_STRUCT && target->sdef &&
                    sig && sig->ret.type.kind == TY_STRUCT &&
                    sig->ret.sdef == target->sdef) {
                    // Pass &q as the sret slot; no extra copy needed.
                    emit_flat_call(e, sc, sig, ex->rhs_assign->args, NULL, sym_addr(e, target));
                    return r;
                }
            }
            LVal lv = emit_lvalue(e, sc, ex->lvalue);
            EVal v = emit_expr(e, sc, ex->rhs_assign);
            v.val = coerce_eval(e, v, lv.elem_ty);
            v.is_float = (lv.elem_ty == e->f32);
            v.is_i64 = (lv.elem_ty == e->i64);
            store_lvalue(e, lv, v.val);
            return v;
        }
        case EX_UN: {
            EVal a = emit_expr(e, sc, ex->lhs);
            if (ex->op == OP_NEG) {
                if (a.is_float) {
                    MLIR_ValueHandle z = emit_const_f32(e, 0.0);
                    r.val = emit_binop(e, OP_TYPE_ARITH_SUBF, str_lit("arith.subf"), e->f32, z, a.val);
                    r.is_float = true;
                } else {
                    MLIR_ValueHandle z = emit_const_i32(e, 0);
                    r.val = emit_binop(e, OP_TYPE_ARITH_SUBI, str_lit("arith.subi"), e->i32, z, a.val);
                }
                return r;
            }
            if (ex->op == OP_BNOT) {
                if (a.is_float) {
                    EMIT_ERR(e, "~ not supported on floats");
                    r.val = emit_const_i32(e, 0);
                    return r;
                }
                MLIR_ValueHandle ones = emit_const_i32(e, -1);
                r.val = emit_binop(e, OP_TYPE_ARITH_XORI, str_lit("arith.xori"),
                                   e->i32, a.val, ones);
                return r;
            }
            MLIR_ValueHandle b = emit_to_bool_i1(e, a);
            r.val = emit_select(e, e->i32, b, emit_const_i32(e, 0), emit_const_i32(e, 1));
            return r;
        }
        case EX_BIN: {
            if (ex->op == OP_AND || ex->op == OP_OR) {
                EVal a = emit_expr(e, sc, ex->lhs);
                MLIR_ValueHandle ab = emit_to_bool_i1(e, a);

                MLIR_RegionHandle then_r = MLIR_CreateRegion(e->ctx);
                MLIR_BlockHandle then_b = MLIR_CreateBlock(e->ctx);
                MLIR_AppendRegionBlock(e->ctx, then_r, then_b);
                MLIR_RegionHandle else_r = MLIR_CreateRegion(e->ctx);
                MLIR_BlockHandle else_b = MLIR_CreateBlock(e->ctx);
                MLIR_AppendRegionBlock(e->ctx, else_r, else_b);

                MLIR_ValueHandle if_res = MLIR_CreateValueOpResult(
                    e->ctx, MLIR_INVALID_HANDLE, 0, e->i32, ssa_name(e), eloc(e, 0));
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i32;
                MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = if_res;
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = ab;
                MLIR_RegionHandle *regs = arena_new_array(e->arena, MLIR_RegionHandle, 2);
                regs[0] = then_r; regs[1] = else_r;
                emit_op(e, OP_TYPE_SCF_IF, str_lit("scf.if"),
                        rts, 1, rs, 1, ops, 1, NULL, 0, regs, 2);

                MLIR_BlockHandle saved = e->cur_block;
                bool saved_term = e->terminated;

                e->cur_block = then_b; e->terminated = false;
                MLIR_ValueHandle then_val;
                if (ex->op == OP_AND) {
                    EVal b = emit_expr(e, sc, ex->rhs);
                    then_val = bool_to_i32(e, emit_to_bool_i1(e, b));
                } else {
                    then_val = emit_const_i32(e, 1);
                }
                MLIR_ValueHandle *ty = arena_new_array(e->arena, MLIR_ValueHandle, 1); ty[0] = then_val;
                emit_op(e, OP_TYPE_SCF_YIELD, str_lit("scf.yield"),
                        NULL, 0, NULL, 0, ty, 1, NULL, 0, NULL, 0);

                e->cur_block = else_b; e->terminated = false;
                MLIR_ValueHandle else_val;
                if (ex->op == OP_AND) {
                    else_val = emit_const_i32(e, 0);
                } else {
                    EVal b = emit_expr(e, sc, ex->rhs);
                    else_val = bool_to_i32(e, emit_to_bool_i1(e, b));
                }
                MLIR_ValueHandle *ey = arena_new_array(e->arena, MLIR_ValueHandle, 1); ey[0] = else_val;
                emit_op(e, OP_TYPE_SCF_YIELD, str_lit("scf.yield"),
                        NULL, 0, NULL, 0, ey, 1, NULL, 0, NULL, 0);
                e->cur_block = saved; e->terminated = saved_term;
                r.val = if_res;
                return r;
            }
            EVal a = emit_expr(e, sc, ex->lhs);
            EVal b = emit_expr(e, sc, ex->rhs);
            // Pointer comparisons (eq/ne) use llvm.icmp on !llvm.ptr.
            if ((a.is_ptr || b.is_ptr) &&
                (ex->op == OP_EQ || ex->op == OP_NE)) {
                MLIR_ValueHandle av = a.val, bv = b.val;
                if (!a.is_ptr) av = emit_null_ptr(e);  // can only be EX_NULL
                if (!b.is_ptr) bv = emit_null_ptr(e);
                int64_t pred = (ex->op == OP_EQ) ? 0 : 1;
                MLIR_ValueHandle cmp = emit_icmp_ptr(e, pred, av, bv);
                r.val = bool_to_i32(e, cmp);
                return r;
            }
            // Pointer ordering comparisons (<, <=, >, >=) on two pointers
            // use llvm.icmp with signed-less-than predicates.
            if (a.is_ptr && b.is_ptr &&
                (ex->op == OP_LT || ex->op == OP_LE ||
                 ex->op == OP_GT || ex->op == OP_GE)) {
                MLIR_ValueHandle cmp = emit_icmp_ptr(e, cmpi_pred_for(ex->op), a.val, b.val);
                r.val = bool_to_i32(e, cmp);
                return r;
            }
            // Pointer arithmetic: p + i, i + p (-> !llvm.ptr via GEP).
            if (ex->op == OP_ADD && (a.is_ptr || b.is_ptr) && !(a.is_ptr && b.is_ptr)) {
                if (a.is_void_ptr || b.is_void_ptr) {
                    EMIT_ERR(e, "pointer arithmetic on 'void*' is not allowed");
                }
                EVal pv = a.is_ptr ? a : b;
                EVal iv = a.is_ptr ? b : a;
                MLIR_TypeHandle elem = (pv.ptr_elem != MLIR_INVALID_HANDLE) ? pv.ptr_elem : e->i32;
                MLIR_ValueHandle idx = iv.is_float ? emit_const_i32(e, 0) : iv.val;
                int32_t *path = arena_new_array(e->arena, int32_t, 1); path[0] = LLVM_GEP_DYN;
                MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, 1); dyn[0] = idx;
                r.val = emit_gep(e, pv.val, elem, path, 1, dyn, 1);
                r.is_ptr = true;
                r.ptr_elem = elem;
                r.is_str = pv.is_str;
                return r;
            }
            // Pointer minus integer: p - i -> GEP %p[-i].
            if (ex->op == OP_SUB && a.is_ptr && !b.is_ptr) {
                if (a.is_void_ptr) {
                    EMIT_ERR(e, "pointer arithmetic on 'void*' is not allowed");
                }
                MLIR_TypeHandle elem = (a.ptr_elem != MLIR_INVALID_HANDLE) ? a.ptr_elem : e->i32;
                MLIR_ValueHandle zero = emit_const_i32(e, 0);
                MLIR_ValueHandle neg = emit_binop(e, OP_TYPE_ARITH_SUBI,
                                                  str_lit("arith.subi"), e->i32, zero, b.val);
                int32_t *path = arena_new_array(e->arena, int32_t, 1); path[0] = LLVM_GEP_DYN;
                MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, 1); dyn[0] = neg;
                r.val = emit_gep(e, a.val, elem, path, 1, dyn, 1);
                r.is_ptr = true;
                r.ptr_elem = elem;
                r.is_str = a.is_str;
                return r;
            }
            // Pointer minus pointer (T* - T*) -> i32 element count.
            if (ex->op == OP_SUB && a.is_ptr && b.is_ptr) {
                if (a.is_void_ptr || b.is_void_ptr) {
                    EMIT_ERR(e, "pointer arithmetic on 'void*' is not allowed");
                }
                MLIR_TypeHandle elem = (a.ptr_elem != MLIR_INVALID_HANDLE) ? a.ptr_elem
                                       : (b.ptr_elem != MLIR_INVALID_HANDLE) ? b.ptr_elem
                                       : e->i32;
                // ptrtoint both, then (ai - bi) / sizeof(elem), trunc to i32.
                MLIR_ValueHandle ai = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                               e->i64, ssa_name(e), eloc(e, 0));
                MLIR_TypeHandle *rts1 = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts1[0] = e->i64;
                MLIR_ValueHandle *rs1 = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs1[0] = ai;
                MLIR_ValueHandle *ops1 = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops1[0] = a.val;
                emit_op(e, OP_TYPE_LLVM_PTRTOINT, str_lit("llvm.ptrtoint"),
                        rts1, 1, rs1, 1, ops1, 1, NULL, 0, NULL, 0);
                MLIR_ValueHandle bi = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                               e->i64, ssa_name(e), eloc(e, 0));
                MLIR_TypeHandle *rts2 = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts2[0] = e->i64;
                MLIR_ValueHandle *rs2 = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs2[0] = bi;
                MLIR_ValueHandle *ops2 = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops2[0] = b.val;
                emit_op(e, OP_TYPE_LLVM_PTRTOINT, str_lit("llvm.ptrtoint"),
                        rts2, 1, rs2, 1, ops2, 1, NULL, 0, NULL, 0);
                MLIR_ValueHandle diff = emit_binop(e, OP_TYPE_ARITH_SUBI,
                                                   str_lit("arith.subi"), e->i64, ai, bi);
                int64_t es = 4;
                if (elem == e->i8) es = 1; else if (elem == e->f32) es = 4;
                else if (elem == e->ptr) es = 8;
                MLIR_ValueHandle szc = emit_const_i64(e, es);
                MLIR_ValueHandle q = emit_binop(e, OP_TYPE_ARITH_DIVSI,
                                                 str_lit("arith.divsi"), e->i64, diff, szc);
                // Truncate to i32 for tinyc's `int` result type.
                MLIR_ValueHandle res = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                                e->i32, ssa_name(e), eloc(e, 0));
                MLIR_TypeHandle *rts3 = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts3[0] = e->i32;
                MLIR_ValueHandle *rs3 = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs3[0] = res;
                MLIR_ValueHandle *ops3 = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops3[0] = q;
                emit_op(e, OP_TYPE_ARITH_TRUNCI, str_lit("arith.trunci"),
                        rts3, 1, rs3, 1, ops3, 1, NULL, 0, NULL, 0);
                r.val = res;
                return r;
            }
            unify_numeric(e, &a, &b);
            bool flt = a.is_float;
            bool i64m = a.is_i64;
            MLIR_TypeHandle ity = i64m ? e->i64 : e->i32;
            switch (ex->op) {
                case OP_ADD:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_ADDF, str_lit("arith.addf"), e->f32, a.val, b.val)
                                : emit_binop(e, OP_TYPE_ARITH_ADDI, str_lit("arith.addi"), ity, a.val, b.val);
                    r.is_float = flt; r.is_i64 = i64m && !flt; return r;
                case OP_SUB:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_SUBF, str_lit("arith.subf"), e->f32, a.val, b.val)
                                : emit_binop(e, OP_TYPE_ARITH_SUBI, str_lit("arith.subi"), ity, a.val, b.val);
                    r.is_float = flt; r.is_i64 = i64m && !flt; return r;
                case OP_MUL:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_MULF, str_lit("arith.mulf"), e->f32, a.val, b.val)
                                : emit_binop(e, OP_TYPE_ARITH_MULI, str_lit("arith.muli"), ity, a.val, b.val);
                    r.is_float = flt; r.is_i64 = i64m && !flt; return r;
                case OP_DIV:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_DIVF, str_lit("arith.divf"), e->f32, a.val, b.val)
                                : emit_binop(e, OP_TYPE_ARITH_DIVSI, str_lit("arith.divsi"), ity, a.val, b.val);
                    r.is_float = flt; r.is_i64 = i64m && !flt; return r;
                case OP_MOD:
                    if (flt) {
                        EMIT_ERR(e, "% not supported on floats");
                        r.val = emit_const_f32(e, 0.0); r.is_float = true; return r;
                    }
                    r.val = emit_binop(e, OP_TYPE_ARITH_REMSI, str_lit("arith.remsi"), ity, a.val, b.val);
                    r.is_i64 = i64m; return r;
                case OP_LT: case OP_LE: case OP_GT: case OP_GE:
                case OP_EQ: case OP_NE: {
                    MLIR_ValueHandle cmp = flt
                        ? emit_cmpf(e, cmpf_pred_for(ex->op), a.val, b.val)
                        : emit_cmpi(e, cmpi_pred_for(ex->op), a.val, b.val);
                    r.val = bool_to_i32(e, cmp);
                    return r;
                }
                case OP_BAND:
                    if (flt) { EMIT_ERR(e, "& not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = emit_binop(e, OP_TYPE_ARITH_ANDI, str_lit("arith.andi"), ity, a.val, b.val);
                    r.is_i64 = i64m; return r;
                case OP_BOR:
                    if (flt) { EMIT_ERR(e, "| not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = emit_binop(e, OP_TYPE_ARITH_ORI, str_lit("arith.ori"), ity, a.val, b.val);
                    r.is_i64 = i64m; return r;
                case OP_BXOR:
                    if (flt) { EMIT_ERR(e, "^ not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = emit_binop(e, OP_TYPE_ARITH_XORI, str_lit("arith.xori"), ity, a.val, b.val);
                    r.is_i64 = i64m; return r;
                case OP_SHL:
                    if (flt) { EMIT_ERR(e, "<< not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = emit_binop(e, OP_TYPE_ARITH_SHLI, str_lit("arith.shli"), ity, a.val, b.val);
                    r.is_i64 = i64m; return r;
                case OP_SHR:
                    if (flt) { EMIT_ERR(e, ">> not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = emit_binop(e, OP_TYPE_ARITH_SHRSI, str_lit("arith.shrsi"), ity, a.val, b.val);
                    r.is_i64 = i64m; return r;
                default: break;
            }
            r.val = emit_const_i32(e, 0);
            return r;
        }
        case EX_CALL: {
            // Indirect call through a callable expression captured in
            // `ex->lhs` (e.g. `(*fp)(args)` produced by parse_postfix). The
            // expression must yield an EVal with is_ptr=true and a
            // fnptr_ty signature. We branch into the same call_indirect
            // emission used for the named-fnptr-variable case below.
            Type *indirect_fnty = NULL;
            MLIR_ValueHandle indirect_callee_ptr = MLIR_INVALID_HANDLE;
            if (ex->callee.size == 0 && ex->lhs) {
                EVal cv = emit_expr(e, sc, ex->lhs);
                if (!cv.is_ptr || !cv.fnptr_ty || !cv.fnptr_ty->fnptr_ret) {
                    EMIT_ERR(e, "callee expression is not a function pointer");
                    r.val = emit_const_i32(e, 0);
                    return r;
                }
                indirect_fnty = cv.fnptr_ty;
                indirect_callee_ptr = cv.val;
            }
            // Built-in va_start/va_end/va_copy: lower to LLVM intrinsics
            // operating on the !llvm.ptr that backs the va_list buffer.
            // The first argument must be a va_list lvalue; we take its
            // address (the alloca pointer) directly.
            if (!indirect_fnty && (str_eq(ex->callee, str_lit("va_start"))
                                 || str_eq(ex->callee, str_lit("va_end"))
                                 || str_eq(ex->callee, str_lit("va_copy")))) {
                bool is_start = str_eq(ex->callee, str_lit("va_start"));
                bool is_copy  = str_eq(ex->callee, str_lit("va_copy"));
                size_t expected = is_start ? 2 : (is_copy ? 2 : 1);
                if (ex->args.size != expected) {
                    EMIT_ERR(e, "{} expects {} argument(s)", ex->callee, (int64_t)expected);
                    r.val = emit_const_i32(e, 0); return r;
                }
                Expr *ap = ex->args.data[0];
                if (ap->kind != EX_VAR) {
                    EMIT_ERR(e, "{} first argument must be a va_list variable", ex->callee);
                    r.val = emit_const_i32(e, 0); return r;
                }
                Sym *sy = lookup(e, sc, ap->name);
                if (!sy || sy->type.kind != TY_VA_LIST) {
                    EMIT_ERR(e, "{} first argument must be of type va_list", ex->callee);
                    r.val = emit_const_i32(e, 0); return r;
                }
                MLIR_ValueHandle ap_ptr = sym_addr(e, sy);
                if (is_copy) {
                    Expr *src = ex->args.data[1];
                    if (src->kind != EX_VAR) {
                        EMIT_ERR(e, "va_copy second argument must be a va_list variable");
                        r.val = emit_const_i32(e, 0); return r;
                    }
                    Sym *ssy = lookup(e, sc, src->name);
                    if (!ssy || ssy->type.kind != TY_VA_LIST) {
                        EMIT_ERR(e, "va_copy second argument must be of type va_list");
                        r.val = emit_const_i32(e, 0); return r;
                    }
                    MLIR_ValueHandle src_ptr = sym_addr(e, ssy);
                    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 2);
                    ops[0] = ap_ptr; ops[1] = src_ptr;
                    emit_op(e, OP_TYPE_UNREGISTERED, str_lit("llvm.intr.vacopy"),
                            NULL, 0, NULL, 0, ops, 2, NULL, 0, NULL, 0);
                } else {
                    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                    ops[0] = ap_ptr;
                    string opname = is_start ? str_lit("llvm.intr.vastart")
                                             : str_lit("llvm.intr.vaend");
                    emit_op(e, OP_TYPE_UNREGISTERED, opname,
                            NULL, 0, NULL, 0, ops, 1, NULL, 0, NULL, 0);
                }
                r.val = emit_const_i32(e, 0);
                return r;
            }
            // Built-in malloc(size) -> !llvm.ptr; size is i32 (typically
            // sizeof), extended to i64 for the libc signature.
            if (!indirect_fnty && str_eq(ex->callee, str_lit("malloc"))) {                if (ex->args.size != 1) {
                    EMIT_ERR(e, "malloc expects 1 argument");
                    r.val = emit_null_ptr(e); r.is_ptr = true; return r;
                }
                MLIR_ValueHandle sz_i32 = emit_expr_i32(e, sc, ex->args.data[0]);
                MLIR_ValueHandle sz_i64 = emit_extsi_i32_to_i64(e, sz_i32);
                MLIR_ValueHandle res = MLIR_CreateValueOpResult(
                    e->ctx, MLIR_INVALID_HANDLE, 0, e->ptr, ssa_name(e), eloc(e, 0));
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->ptr;
                MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = res;
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = sz_i64;
                MLIR_AttributeHandle ca = MLIR_CreateAttributeSymbolRef(
                    e->ctx, str_lit("callee"), str_lit("malloc"));
                MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = ca;
                emit_op(e, OP_TYPE_FUNC_CALL, str_lit("func.call"),
                        rts, 1, rs, 1, ops, 1, as, 1, NULL, 0);
                r.val = res; r.is_ptr = true; return r;
            }
            if (!indirect_fnty && str_eq(ex->callee, str_lit("free"))) {
                if (ex->args.size != 1) {
                    EMIT_ERR(e, "free expects 1 argument");
                    return r;
                }
                EVal pv = emit_expr(e, sc, ex->args.data[0]);
                MLIR_ValueHandle p = pv.is_ptr ? pv.val : emit_null_ptr(e);
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = p;
                MLIR_AttributeHandle ca = MLIR_CreateAttributeSymbolRef(
                    e->ctx, str_lit("callee"), str_lit("free"));
                MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = ca;
                emit_op(e, OP_TYPE_FUNC_CALL, str_lit("func.call"),
                        NULL, 0, NULL, 0, ops, 1, as, 1, NULL, 0);
                r.val = emit_const_i32(e, 0);
                return r;
            }
            FuncSig *sig = indirect_fnty ? NULL : find_sig(e, ex->callee);
            if (!sig) {
                Type *fnty = NULL;
                MLIR_ValueHandle callee_ptr = MLIR_INVALID_HANDLE;
                if (indirect_fnty) {
                    fnty = indirect_fnty;
                    callee_ptr = indirect_callee_ptr;
                } else {
                    // Indirect call through a function-pointer value bound to
                    // `ex->callee` (local var, parameter, or module global).
                    Sym *sy = lookup(e, sc, ex->callee);
                    if (!sy || sy->type.kind != TY_FNPTR || !sy->type.fnptr_ret) {
                        EMIT_ERR(e, "undefined function: {}", ex->callee);
                        r.val = emit_const_i32(e, 0);
                        return r;
                    }
                    fnty = &sy->type;
                    // Load the !llvm.ptr from the variable's storage.
                    callee_ptr = emit_load_v(e, sym_addr(e, sy), e->ptr);
                }
                size_t na = ex->args.size;
                if ((int)na != fnty->fnptr_nparams) {
                    EMIT_ERR(e, "indirect call to {} arity mismatch",
                            ex->callee);
                }
                // Build the function MLIR type that func.call_indirect needs.
                size_t np = (size_t)fnty->fnptr_nparams;
                MLIR_TypeHandle *in_tys = arena_new_array(e->arena, MLIR_TypeHandle,
                                                           np ? np : 1);
                for (size_t i = 0; i < np; i++)
                    in_tys[i] = scalar_mlir_type(e, fnty->fnptr_params[i].kind);
                MLIR_TypeHandle rty = scalar_mlir_type(e, fnty->fnptr_ret->kind);
                MLIR_TypeHandle out_tys[1] = { rty };
                MLIR_TypeHandle f_ty = MLIR_CreateTypeFunction(e->ctx,
                    in_tys, np, out_tys, 1);
                // Cast !llvm.ptr -> function type so func.call_indirect's
                // verifier is happy. The reconcile-unrealized-casts pass
                // eliminates this after func-to-llvm lowering.
                MLIR_ValueHandle callee_v = MLIR_CreateValueOpResult(
                    e->ctx, MLIR_INVALID_HANDLE, 0, f_ty, ssa_name(e), eloc(e, 0));
                MLIR_TypeHandle *crts = arena_new_array(e->arena, MLIR_TypeHandle, 1);
                crts[0] = f_ty;
                MLIR_ValueHandle *crs = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                crs[0] = callee_v;
                MLIR_ValueHandle *cops = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                cops[0] = callee_ptr;
                emit_op(e, OP_TYPE_UNREALIZED_CONVERSION_CAST,
                        str_lit("builtin.unrealized_conversion_cast"),
                        crts, 1, crs, 1, cops, 1, NULL, 0, NULL, 0);

                size_t n_ops = 1 + na;
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle,
                                                         n_ops ? n_ops : 1);
                ops[0] = callee_v;
                for (size_t i = 0; i < na; i++) {
                    EVal av = emit_expr(e, sc, ex->args.data[i]);
                    TypeKind want = (i < (size_t)fnty->fnptr_nparams)
                        ? fnty->fnptr_params[i].kind : TY_I32;
                    ops[1 + i] = coerce_eval(e, av, scalar_mlir_type(e, want));
                }
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1);
                MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                rts[0] = rty;
                rs[0] = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  rty, ssa_name(e), eloc(e, 0));
                emit_op(e, OP_TYPE_FUNC_CALL_INDIRECT, str_lit("func.call_indirect"),
                        rts, 1, rs, 1, ops, n_ops, NULL, 0, NULL, 0);
                r.val = rs[0];
                r.is_float = (fnty->fnptr_ret->kind == TY_F32);
                r.is_ptr = (rty == e->ptr);
                return r;
            }
            if (sig->ret.type.kind == TY_STRUCT) {
                MLIR_TypeHandle st = find_struct_type(e, sig->ret.sdef);
                MLIR_ValueHandle tmp = emit_alloca(e, st);
                emit_flat_call(e, sc, sig, ex->args, NULL, tmp);
                r.val = tmp;
                r.is_ptr = true;
                r.sdef = sig->ret.sdef;
                return r;
            }
            if (sig->ret.type.kind == TY_VOID) {
                EMIT_ERR(e, "void-returning call has no value (use as a statement)");
                emit_flat_call(e, sc, sig, ex->args, NULL, MLIR_INVALID_HANDLE);
                r.val = emit_const_i32(e, 0);
                return r;
            }
            MLIR_ValueHandle *results = arena_new_array(e->arena, MLIR_ValueHandle, 1);
            emit_flat_call(e, sc, sig, ex->args, results, MLIR_INVALID_HANDLE);
            r.val = results[0];
            r.is_float = (sig->ret.type.kind == TY_F32);
            r.is_i64 = (sig->ret.type.kind == TY_I64);
            if (sig->ret.type.kind == TY_PTR_STRUCT) {
                r.is_ptr = true;
                r.sdef = sig->ret.sdef;
            }
            if (sig->ret.type.kind == TY_PTR_I32 ||
                sig->ret.type.kind == TY_PTR_CHAR ||
                sig->ret.type.kind == TY_PTR_VOID ||
                sig->ret.type.kind == TY_FNPTR) {
                r.is_ptr = true;
            }
            return r;
        }
        case EX_COMPOUND: {
            EVal r = (EVal){0};
            Type t = ex->cast_type;
            if (t.kind == TY_STRUCT) {
                StructDef *sd = find_struct(e, t.struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct in compound literal");
                    return r;
                }
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                MLIR_ValueHandle addr = emit_alloca(e, st_ty);
                int32_t prefix0[1] = {0};
                emit_struct_zero(e, addr, st_ty, sd, prefix0, 1);
                size_t n = ex->args.size;
                if (n > sd->fields.size) {
                    EMIT_ERR(e, "too many initializers in compound literal");
                    n = sd->fields.size;
                }
                for (size_t i = 0; i < n; i++) {
                    Type ft = sd->fields.data[i].type;
                    int32_t path[2] = {0, (int32_t)i};
                    MLIR_ValueHandle p =
                        emit_gep(e, addr, st_ty, path, 2, NULL, 0);
                    EVal v = emit_expr(e, sc, ex->args.data[i]);
                    MLIR_TypeHandle want = scalar_mlir_type(e, ft.kind);
                    MLIR_ValueHandle sv;
                    if (ft.kind == TY_PTR_I32 || ft.kind == TY_PTR_CHAR ||
                        ft.kind == TY_PTR_VOID || ft.kind == TY_PTR_STRUCT ||
                        ft.kind == TY_FNPTR || ft.kind == TY_PTR_PTR) {
                        sv = v.val;
                    } else {
                        sv = coerce_eval(e, v, want);
                    }
                    emit_store_v(e, sv, p);
                }
                r.val = addr;
                r.is_ptr = true;
                r.sdef = sd;
                return r;
            }
            if (ex->args.size != 1) {
                EMIT_ERR(e, "scalar compound literal needs exactly one value");
                return r;
            }
            EVal v = emit_expr(e, sc, ex->args.data[0]);
            if (t.kind == TY_I32) {
                r.val = coerce_eval(e, v, e->i32);
                return r;
            }
            if (t.kind == TY_I64) {
                r.val = coerce_eval(e, v, e->i64);
                r.is_i64 = true;
                return r;
            }
            if (t.kind == TY_F32) {
                r.val = coerce_eval(e, v, e->f32);
                r.is_float = true;
                return r;
            }
            if (t.kind == TY_PTR_I32 || t.kind == TY_PTR_CHAR ||
                t.kind == TY_PTR_VOID || t.kind == TY_PTR_STRUCT ||
                t.kind == TY_PTR_PTR || t.kind == TY_FNPTR) {
                r = v;
                r.is_ptr = true;
                r.is_void_ptr = (t.kind == TY_PTR_VOID);
                if (t.kind == TY_PTR_STRUCT) {
                    r.sdef = find_struct(e, t.struct_name);
                }
                return r;
            }
            EMIT_ERR(e, "unsupported type in compound literal");
            return r;
        }
    }
    r.val = emit_const_i32(e, 0);
    return r;
}

static void emit_block(E *e, Scope *sc, VecStmtPtr body);

// Zero-initialize a struct local field-by-field.
static void emit_struct_zero(E *e, MLIR_ValueHandle base, MLIR_TypeHandle source_elem,
                             StructDef *sd, int32_t *prefix, size_t n_prefix) {
    for (size_t i = 0; i < sd->fields.size; i++) {
        Type ft = sd->fields.data[i].type;
        size_t n_path = n_prefix + 1;
        int32_t *path = arena_new_array(e->arena, int32_t, n_path);
        for (size_t k = 0; k < n_prefix; k++) path[k] = prefix[k];
        path[n_prefix] = (int32_t)i;
        if (ft.kind == TY_STRUCT) {
            StructDef *inner = find_struct(e, ft.struct_name);
            emit_struct_zero(e, base, source_elem, inner, path, n_path);
        } else if (ft.kind == TY_PTR_STRUCT || ft.kind == TY_PTR_I32 ||
                   ft.kind == TY_PTR_CHAR || ft.kind == TY_PTR_VOID) {
            MLIR_ValueHandle p = emit_gep(e, base, source_elem, path, n_path, NULL, 0);
            emit_store_v(e, emit_null_ptr(e), p);
        } else if (ft.kind == TY_ARRAY_I32 || ft.kind == TY_ARRAY_F32 ||
                   ft.kind == TY_ARRAY_PTR_STRUCT || ft.kind == TY_ARRAY_PTR_CHAR) {
            // Zero each element with explicit GEPs. Cheap and avoids
            // reaching for llvm.memset.
            bool is_2d = (ft.array_len2 != 0);
            int64_t n1 = ft.array_len;
            int64_t n2 = is_2d ? ft.array_len2 : 1;
            MLIR_ValueHandle z;
            if (ft.kind == TY_ARRAY_F32) { z = emit_const_f32(e, 0.0); }
            else if (ft.kind == TY_ARRAY_PTR_STRUCT || ft.kind == TY_ARRAY_PTR_CHAR) {
                z = emit_null_ptr(e);
            } else { z = emit_const_i32(e, 0); }
            for (int64_t a = 0; a < n1; a++) {
                for (int64_t b = 0; b < n2; b++) {
                    size_t inner_n = is_2d ? 2 : 1;
                    size_t total = n_path + inner_n;
                    int32_t *p2 = arena_new_array(e->arena, int32_t, total);
                    for (size_t kk = 0; kk < n_path; kk++) p2[kk] = path[kk];
                    p2[n_path] = (int32_t)a;
                    if (is_2d) p2[n_path + 1] = (int32_t)b;
                    MLIR_ValueHandle p = emit_gep(e, base, source_elem, p2, total, NULL, 0);
                    emit_store_v(e, z, p);
                }
            }
        } else {
            MLIR_ValueHandle p = emit_gep(e, base, source_elem, path, n_path, NULL, 0);
            MLIR_ValueHandle z = (ft.kind == TY_F32) ? emit_const_f32(e, 0.0) : emit_const_i32(e, 0);
            emit_store_v(e, z, p);
        }
    }
}

static void emit_stmt(E *e, Scope *sc, Stmt *st) {
    if (e->terminated) return;
    e->cur_line = st->line;
    switch (st->kind) {
        case ST_EXPR: {
            if (st->expr->kind == EX_CALL) {
                FuncSig *sig = find_sig(e, st->expr->callee);
                if (sig && sig->ret.type.kind == TY_STRUCT) {
                    emit_flat_call(e, sc, sig, st->expr->args, NULL, MLIR_INVALID_HANDLE);
                    return;
                }
                if (sig && sig->ret.type.kind == TY_VOID) {
                    emit_flat_call(e, sc, sig, st->expr->args, NULL, MLIR_INVALID_HANDLE);
                    return;
                }
            }
            (void)emit_expr(e, sc, st->expr);
            return;
        }
        case ST_DECL: {
            Sym *sy = arena_new(e->arena, Sym);
            sy->name = st->decl_name;
            sy->type = st->decl_type;

            if (st->decl_type.kind == TY_VA_LIST) {
                // Allocate a 32-byte buffer (sufficient for x86_64-SysV
                // and aarch64 va_list layouts on Linux/macOS/Windows). We
                // hand out the pointer; va_start/va_arg/va_end consume it.
                MLIR_TypeHandle buf_ty = MLIR_CreateTypeLLVMArray(e->ctx, e->i8, 32);
                sy->addr = emit_alloca(e, buf_ty);
                sy->next = sc->head;
                sc->head = sy;
                break;
            }
            if (st->decl_type.kind == TY_PTR_I32 || st->decl_type.kind == TY_PTR_VOID ||
                st->decl_type.kind == TY_PTR_PTR) {
                sy->addr = emit_alloca(e, e->ptr);
                if (st->decl_init) {
                    EVal iv = emit_expr(e, sc, st->decl_init);
                    if (!iv.is_ptr) {
                        EMIT_ERR(e, "pointer initializer must be a pointer expression");
                    } else {
                        emit_store_v(e, iv.val, sy->addr);
                    }
                }
            } else if (st->decl_type.kind == TY_PTR_CHAR) {
                sy->addr = emit_alloca(e, e->ptr);
                if (st->decl_init) {
                    EVal iv = emit_expr(e, sc, st->decl_init);
                    MLIR_ValueHandle p = iv.is_ptr ? iv.val : emit_null_ptr(e);
                    emit_store_v(e, p, sy->addr);
                } else {
                    emit_store_v(e, emit_null_ptr(e), sy->addr);
                }
            } else if (st->decl_type.kind == TY_FNPTR) {
                sy->addr = emit_alloca(e, e->ptr);
                if (st->decl_init) {
                    EVal iv = emit_expr(e, sc, st->decl_init);
                    MLIR_ValueHandle p = iv.is_ptr ? iv.val : emit_null_ptr(e);
                    emit_store_v(e, p, sy->addr);
                } else {
                    emit_store_v(e, emit_null_ptr(e), sy->addr);
                }
            } else if (st->decl_type.kind == TY_PTR_STRUCT) {
                StructDef *sd = find_struct(e, st->decl_type.struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct type {}", st->decl_type.struct_name);
                    return;
                }
                sy->sdef = sd;
                sy->addr = emit_alloca(e, e->ptr);
                if (st->decl_init) {
                    if (st->decl_init->kind != EX_ADDR ||
                        st->decl_init->lhs->kind != EX_VAR) {
                        EMIT_ERR(e, "struct* must be initialized with &<struct_var>");
                    } else {
                        Sym *target = lookup(e, sc, st->decl_init->lhs->name);
                        if (!target || target->type.kind != TY_STRUCT || target->sdef != sd) {
                            EMIT_ERR(e, "struct* target type mismatch");
                        } else {
                            emit_store_v(e, sym_addr(e, target), sy->addr);
                        }
                    }
                }
            } else if (st->decl_type.kind == TY_ARRAY_I32) {
                MLIR_TypeHandle arr_ty;
                if (st->decl_type.array_len2 != 0) {
                    MLIR_TypeHandle inner = MLIR_CreateTypeLLVMArray(
                        e->ctx, e->i32, (uint64_t)st->decl_type.array_len2);
                    arr_ty = MLIR_CreateTypeLLVMArray(
                        e->ctx, inner, (uint64_t)st->decl_type.array_len);
                } else {
                    arr_ty = MLIR_CreateTypeLLVMArray(
                        e->ctx, e->i32, (uint64_t)st->decl_type.array_len);
                }
                sy->addr = emit_alloca(e, arr_ty);
                if (st->decl_init) {
                    EMIT_ERR(e, "array initializers are not supported");
                }
            } else if (st->decl_type.kind == TY_ARRAY_STRUCT) {
                StructDef *sd = find_struct(e, st->decl_type.struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct type {}", st->decl_type.struct_name);
                    return;
                }
                sy->sdef = sd;
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                MLIR_TypeHandle arr_ty = MLIR_CreateTypeLLVMArray(
                    e->ctx, st_ty, (uint64_t)st->decl_type.array_len);
                sy->addr = emit_alloca(e, arr_ty);
                if (st->decl_init) {
                    EMIT_ERR(e, "array-of-struct initializers are not supported");
                }
            } else if (st->decl_type.kind == TY_STRUCT) {
                StructDef *sd = find_struct(e, st->decl_type.struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct type {}", st->decl_type.struct_name);
                    return;
                }
                sy->sdef = sd;
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                sy->addr = emit_alloca(e, st_ty);
                int32_t *prefix = arena_new_array(e->arena, int32_t, 1);
                prefix[0] = 0;
                emit_struct_zero(e, sy->addr, st_ty, sd, prefix, 1);
                if (st->decl_init) {
                    if (st->decl_init->kind != EX_COMPOUND ||
                        st->decl_init->cast_type.kind != TY_STRUCT ||
                        !str_eq(st->decl_init->cast_type.struct_name,
                                st->decl_type.struct_name)) {
                        EMIT_ERR(e, "struct initializer must be a "
                                    "matching compound literal");
                    } else {
                        size_t n = st->decl_init->args.size;
                        if (n > sd->fields.size) {
                            EMIT_ERR(e, "too many initializers");
                            n = sd->fields.size;
                        }
                        for (size_t i = 0; i < n; i++) {
                            // Map positional or designated initializer
                            // entry to the target field index. For
                            // designated entries (.fname = v), look up
                            // fname in the struct's field list.
                            int fidx = (int)i;
                            if (st->decl_init->compound_field_names) {
                                string fname = st->decl_init->compound_field_names[i];
                                if (fname.size != 0) {
                                    fidx = struct_field_index(sd, fname);
                                    if (fidx < 0) {
                                        EMIT_ERR(e, "unknown struct field {} in initializer", fname);
                                        continue;
                                    }
                                }
                            }
                            Type ft = sd->fields.data[fidx].type;
                            int32_t path[2] = {0, (int32_t)fidx};
                            MLIR_ValueHandle p = emit_gep(
                                e, sy->addr, st_ty, path, 2, NULL, 0);
                            EVal v = emit_expr(
                                e, sc, st->decl_init->args.data[i]);
                            MLIR_ValueHandle sv;
                            if (ft.kind == TY_PTR_I32 ||
                                ft.kind == TY_PTR_CHAR ||
                                ft.kind == TY_PTR_VOID ||
                                ft.kind == TY_PTR_STRUCT ||
                                ft.kind == TY_FNPTR ||
                                ft.kind == TY_PTR_PTR) {
                                sv = v.val;
                            } else {
                                sv = coerce_eval(
                                    e, v, scalar_mlir_type(e, ft.kind));
                            }
                            emit_store_v(e, sv, p);
                        }
                    }
                }
            } else if (st->decl_type.kind == TY_I64) {
                sy->addr = emit_alloca(e, e->i64);
                if (st->decl_init) {
                    EVal v = emit_expr(e, sc, st->decl_init);
                    MLIR_ValueHandle iv = coerce_eval(e, v, e->i64);
                    emit_store_v(e, iv, sy->addr);
                } else {
                    emit_store_v(e, emit_const_i64(e, 0), sy->addr);
                }
            } else if (st->decl_type.kind == TY_F32) {
                sy->addr = emit_alloca(e, e->f32);
                if (st->decl_init) {
                    EVal v = emit_expr(e, sc, st->decl_init);
                    if (!v.is_float) { v.val = emit_sitofp(e, v.val); v.is_float = true; }
                    emit_store_v(e, v.val, sy->addr);
                } else {
                    emit_store_v(e, emit_const_f32(e, 0.0), sy->addr);
                }
            } else {
                sy->addr = emit_alloca(e, e->i32);
                if (st->decl_init) {
                    MLIR_ValueHandle v = emit_expr_i32(e, sc, st->decl_init);
                    emit_store_v(e, v, sy->addr);
                } else {
                    emit_store_v(e, emit_const_i32(e, 0), sy->addr);
                }
            }
            sy->next = sc->head;
            sc->head = sy;
            return;
        }
        case ST_RETURN: {
            FuncSig *sig = e->cur_sig;
            bool is_void_fn = (sig && sig->ret.type.kind == TY_VOID);
            if (is_void_fn) {
                if (st->expr) {
                    EMIT_ERR(e, "void function cannot return a value");
                }
                emit_return_op(e, NULL, 0);
                e->terminated = true;
                return;
            }
            if (!st->expr) {
                EMIT_ERR(e, "non-void function requires a return value");
                emit_return_op(e, NULL, 0);
                e->terminated = true;
                return;
            }
            if (sig && sig->ret.type.kind == TY_STRUCT) {
                StructDef *want = sig->ret.sdef;
                MLIR_ValueHandle out = e->cur_sret_ptr;
                if (st->expr->kind == EX_CALL) {
                    FuncSig *csig = find_sig(e, st->expr->callee);
                    if (!csig || csig->ret.type.kind != TY_STRUCT || csig->ret.sdef != want) {
                        EMIT_ERR(e, "returned call type mismatch");
                    } else {
                        emit_flat_call(e, sc, csig, st->expr->args, NULL, out);
                    }
                } else if (st->expr->kind == EX_VAR) {
                    Sym *s = lookup(e, sc, st->expr->name);
                    if (!s || s->type.kind != TY_STRUCT || s->sdef != want) {
                        EMIT_ERR(e, "returned struct type mismatch");
                    } else {
                        emit_struct_copy(e, out, sym_addr(e, s), want);
                    }
                } else {
                    EMIT_ERR(e, "struct return must be a variable or struct call");
                }
                emit_return_op(e, NULL, 0);
                e->terminated = true;
                return;
            }
            MLIR_TypeHandle want_ty;
            if (sig && sig->ret.type.kind == TY_F32) want_ty = e->f32;
            else if (sig && sig->ret.type.kind == TY_I64) want_ty = e->i64;
            else if (sig && (sig->ret.type.kind == TY_PTR_STRUCT ||
                             sig->ret.type.kind == TY_PTR_I32 ||
                             sig->ret.type.kind == TY_PTR_CHAR ||
                             sig->ret.type.kind == TY_PTR_VOID ||
                             sig->ret.type.kind == TY_FNPTR ||
                             sig->ret.type.kind == TY_PTR_PTR)) want_ty = e->ptr;
            else want_ty = e->i32;
            EVal v = emit_expr(e, sc, st->expr);
            MLIR_ValueHandle ret_v;
            if (want_ty == e->ptr) {
                if (!v.is_ptr) {
                    EMIT_ERR(e, "return expects a pointer value");
                    ret_v = emit_null_ptr(e);
                } else {
                    ret_v = v.val;
                }
            } else {
                ret_v = coerce_eval(e, v, want_ty);
            }
            MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = ret_v;
            emit_return_op(e, ops, 1);
            e->terminated = true;
            return;
        }
        case ST_BREAK: {
            if (!e->loops) { EMIT_ERR(e, "break outside of loop"); return; }
            emit_branch(e, e->loops->break_block);
            return;
        }
        case ST_CONTINUE: {
            if (!e->loops) { EMIT_ERR(e, "continue outside of loop"); return; }
            emit_branch(e, e->loops->continue_block);
            return;
        }
        case ST_PRINT: {
            EVal v = emit_expr(e, sc, st->expr);
            if (v.is_str) {
                // print(<string>) -> @printStr(!llvm.ptr)
                e->use_print_str = true;
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                ops[0] = v.val;
                MLIR_AttributeHandle ca = MLIR_CreateAttributeSymbolRef(
                    e->ctx, str_lit("callee"), str_lit("printStr"));
                MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = ca;
                emit_op(e, OP_TYPE_FUNC_CALL, str_lit("func.call"),
                        NULL, 0, NULL, 0, ops, 1, as, 1, NULL, 0);
                return;
            }
            MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1);
            ops[0] = v.val;
            emit_op(e, OP_TYPE_VECTOR_PRINT, str_lit("vector.print"),
                    NULL, 0, NULL, 0, ops, 1, NULL, 0, NULL, 0);
            return;
        }
        case ST_BLOCK: {
            Scope inner = (Scope){.head = NULL, .parent = sc};
            emit_block(e, &inner, st->block_body);
            return;
        }
        case ST_IF: {
            EVal c = emit_expr(e, sc, st->cond);
            MLIR_ValueHandle cb = emit_to_bool_i1(e, c);

            MLIR_BlockHandle then_b = new_cfg_block(e);
            MLIR_BlockHandle else_b = st->has_else ? new_cfg_block(e) : MLIR_INVALID_HANDLE;
            MLIR_BlockHandle end_b  = new_cfg_block(e);
            emit_cond_branch(e, cb, then_b, st->has_else ? else_b : end_b);

            e->cur_block = then_b; e->terminated = false;
            { Scope inner = (Scope){.head = NULL, .parent = sc};
              for (size_t i = 0; i < st->then_body.size && !e->terminated; i++)
                  emit_stmt(e, &inner, st->then_body.data[i]); }
            emit_branch(e, end_b);

            if (st->has_else) {
                e->cur_block = else_b; e->terminated = false;
                Scope inner = (Scope){.head = NULL, .parent = sc};
                for (size_t i = 0; i < st->else_body.size && !e->terminated; i++)
                    emit_stmt(e, &inner, st->else_body.data[i]);
                emit_branch(e, end_b);
            }

            e->cur_block = end_b; e->terminated = false;
            return;
        }
        case ST_WHILE: {
            MLIR_BlockHandle hdr_b  = new_cfg_block(e);
            MLIR_BlockHandle body_b = new_cfg_block(e);
            MLIR_BlockHandle exit_b = new_cfg_block(e);
            emit_branch(e, hdr_b);

            e->cur_block = hdr_b; e->terminated = false;
            EVal c = emit_expr(e, sc, st->cond);
            MLIR_ValueHandle cb = emit_to_bool_i1(e, c);
            emit_cond_branch(e, cb, body_b, exit_b);

            e->cur_block = body_b; e->terminated = false;
            LoopCtx lc = (LoopCtx){.continue_block = hdr_b, .break_block = exit_b, .parent = e->loops};
            e->loops = &lc;
            { Scope inner = (Scope){.head = NULL, .parent = sc};
              for (size_t i = 0; i < st->while_body.size && !e->terminated; i++)
                  emit_stmt(e, &inner, st->while_body.data[i]); }
            e->loops = lc.parent;
            emit_branch(e, hdr_b);

            e->cur_block = exit_b; e->terminated = false;
            return;
        }
        case ST_DO_WHILE: {
            // do { body } while (cond); — body executes once before the
            // condition is tested. `continue` jumps to the cond-check.
            MLIR_BlockHandle body_b = new_cfg_block(e);
            MLIR_BlockHandle cond_b = new_cfg_block(e);
            MLIR_BlockHandle exit_b = new_cfg_block(e);
            emit_branch(e, body_b);

            e->cur_block = body_b; e->terminated = false;
            LoopCtx lc = (LoopCtx){.continue_block = cond_b, .break_block = exit_b, .parent = e->loops};
            e->loops = &lc;
            { Scope inner = (Scope){.head = NULL, .parent = sc};
              for (size_t i = 0; i < st->while_body.size && !e->terminated; i++)
                  emit_stmt(e, &inner, st->while_body.data[i]); }
            e->loops = lc.parent;
            emit_branch(e, cond_b);

            e->cur_block = cond_b; e->terminated = false;
            EVal c = emit_expr(e, sc, st->cond);
            MLIR_ValueHandle cb = emit_to_bool_i1(e, c);
            emit_cond_branch(e, cb, body_b, exit_b);

            e->cur_block = exit_b; e->terminated = false;
            return;
        }
        case ST_FOR: {
            Scope inner = (Scope){.head = NULL, .parent = sc};
            if (st->for_init) emit_stmt(e, &inner, st->for_init);
            if (e->terminated) return;

            MLIR_BlockHandle hdr_b  = new_cfg_block(e);
            MLIR_BlockHandle body_b = new_cfg_block(e);
            MLIR_BlockHandle step_b = new_cfg_block(e);
            MLIR_BlockHandle exit_b = new_cfg_block(e);
            emit_branch(e, hdr_b);

            e->cur_block = hdr_b; e->terminated = false;
            MLIR_ValueHandle cb;
            if (st->cond) {
                EVal c = emit_expr(e, &inner, st->cond);
                cb = emit_to_bool_i1(e, c);
            } else {
                MLIR_ValueHandle one = emit_const_i32(e, 1);
                cb = emit_cmpi(e, /*ne*/ 1, one, emit_const_i32(e, 0));
            }
            emit_cond_branch(e, cb, body_b, exit_b);

            e->cur_block = body_b; e->terminated = false;
            LoopCtx lc = (LoopCtx){.continue_block = step_b, .break_block = exit_b, .parent = e->loops};
            e->loops = &lc;
            for (size_t i = 0; i < st->for_body.size && !e->terminated; i++)
                emit_stmt(e, &inner, st->for_body.data[i]);
            e->loops = lc.parent;
            emit_branch(e, step_b);

            e->cur_block = step_b; e->terminated = false;
            if (st->for_step) (void)emit_expr(e, &inner, st->for_step);
            emit_branch(e, hdr_b);

            e->cur_block = exit_b; e->terminated = false;
            return;
        }
        case ST_SWITCH: {
            // Lower switch as a chain of cf.cond_br tests followed by case
            // bodies that fall through to the next case unless terminated by
            // a `break`. Multiple SwitchCase entries pointing at the same
            // body_start share one CFG block.
            EVal cv = emit_expr(e, sc, st->cond);
            MLIR_ValueHandle cond_v = cv.val;

            size_t nc = st->switch_cases.size;
            MLIR_BlockHandle exit_b = new_cfg_block(e);

            // Per-case body block, one per unique body_start.
            MLIR_BlockHandle *case_blocks = arena_new_array(e->arena, MLIR_BlockHandle, nc ? nc : 1);
            int64_t *body_start = arena_new_array(e->arena, int64_t, nc ? nc : 1);
            for (size_t i = 0; i < nc; i++) {
                int64_t bs = st->switch_cases.data[i].body_start;
                body_start[i] = bs;
                case_blocks[i] = MLIR_INVALID_HANDLE;
                for (size_t j = 0; j < i; j++) {
                    if (body_start[j] == bs) { case_blocks[i] = case_blocks[j]; break; }
                }
                if (case_blocks[i] == MLIR_INVALID_HANDLE) case_blocks[i] = new_cfg_block(e);
            }
            // Locate default block (if any) and emit the test cascade.
            MLIR_BlockHandle default_b = MLIR_INVALID_HANDLE;
            for (size_t i = 0; i < nc; i++) {
                if (st->switch_cases.data[i].is_default) { default_b = case_blocks[i]; break; }
            }
            for (size_t i = 0; i < nc; i++) {
                SwitchCase *c = &st->switch_cases.data[i];
                if (c->is_default) continue;
                MLIR_ValueHandle val = emit_const_i32(e, c->value);
                MLIR_ValueHandle eq = emit_cmpi(e, /*eq*/ 0, cond_v, val);
                MLIR_BlockHandle next_test = new_cfg_block(e);
                emit_cond_branch(e, eq, case_blocks[i], next_test);
                e->cur_block = next_test; e->terminated = false;
            }
            emit_branch(e, default_b != MLIR_INVALID_HANDLE ? default_b : exit_b);

            // Determine the contiguous stmt range for each case block.
            // Inherit `continue` target from any enclosing loop.
            MLIR_BlockHandle cont_target = e->loops ? e->loops->continue_block : MLIR_INVALID_HANDLE;
            LoopCtx lc = (LoopCtx){.continue_block = cont_target, .break_block = exit_b, .parent = e->loops};
            e->loops = &lc;
            Scope inner = (Scope){.head = NULL, .parent = sc};
            for (size_t i = 0; i < nc; i++) {
                // Skip duplicate entries pointing at a body_start already emitted.
                bool dup = false;
                for (size_t j = 0; j < i; j++) if (body_start[j] == body_start[i]) { dup = true; break; }
                if (dup) continue;
                e->cur_block = case_blocks[i]; e->terminated = false;
                int64_t end = (int64_t)st->block_body.size;
                // Find the next case's body_start after this one.
                for (size_t j = 0; j < nc; j++) {
                    if (body_start[j] > body_start[i] && body_start[j] < end)
                        end = body_start[j];
                }
                for (int64_t k = body_start[i]; k < end && !e->terminated; k++) {
                    emit_stmt(e, &inner, st->block_body.data[k]);
                }
                // Fall through to the next case block in source order, or exit.
                if (!e->terminated) {
                    MLIR_BlockHandle nxt = exit_b;
                    int64_t best = -1;
                    for (size_t j = 0; j < nc; j++) {
                        if (body_start[j] > body_start[i] &&
                            (best < 0 || body_start[j] < best)) {
                            best = body_start[j];
                            nxt = case_blocks[j];
                        }
                    }
                    emit_branch(e, nxt);
                }
            }
            e->loops = lc.parent;
            e->cur_block = exit_b; e->terminated = false;
            return;
        }
    }
}

static void emit_block(E *e, Scope *sc, VecStmtPtr body) {
    for (size_t i = 0; i < body.size && !e->terminated; i++) emit_stmt(e, sc, body.data[i]);
}

// ----- function & module -----

static void slot_resolve(E *e, Type ty, SlotInfo *out) {
    out->type = ty;
    out->sdef = NULL;
    if (ty.kind == TY_STRUCT || ty.kind == TY_PTR_STRUCT) {
        out->sdef = find_struct(e, ty.struct_name);
        if (!out->sdef) {
            EMIT_ERR(e, "unknown struct type {}", ty.struct_name);
        }
    } else if (ty.kind != TY_I32 && ty.kind != TY_I64 && ty.kind != TY_F32 &&
               ty.kind != TY_PTR_I32 && ty.kind != TY_PTR_CHAR &&
               ty.kind != TY_PTR_VOID && ty.kind != TY_VOID &&
               ty.kind != TY_FNPTR && ty.kind != TY_PTR_PTR &&
               ty.kind != TY_VA_LIST) {
        EMIT_ERR(e, "unsupported type in function signature");
    }
}

static MLIR_TypeHandle slot_param_type(E *e, SlotInfo *p) {
    if (p->type.kind == TY_STRUCT || p->type.kind == TY_PTR_STRUCT) return e->ptr;
    return scalar_mlir_type(e, p->type.kind);
}

static void build_signatures(E *e) {
    e->n_sigs = e->program->funcs.size;
    e->sigs = arena_new_array(e->arena, FuncSig, e->n_sigs ? e->n_sigs : 1);
    for (size_t f_i = 0; f_i < e->n_sigs; f_i++) {
        Func *f = e->program->funcs.data[f_i];
        FuncSig *sig = &e->sigs[f_i];
        *sig = (FuncSig){0};
        sig->name = f->name;
        sig->func = f;
        sig->is_variadic = f->is_variadic;
        slot_resolve(e, f->return_type, &sig->ret);
        sig->n_params = f->params.size;
        sig->params = arena_new_array(e->arena, SlotInfo, sig->n_params ? sig->n_params : 1);
        for (size_t i = 0; i < sig->n_params; i++) {
            slot_resolve(e, f->params.data[i].type, &sig->params[i]);
        }
        sig->sret = (sig->ret.type.kind == TY_STRUCT);
        // Note: variadic + sret is allowed. The sret pointer is the first
        // argument, followed by fixed args, then variadic args. The function
        // returns void via llvm.return.
        bool is_void = (sig->ret.type.kind == TY_VOID);
        size_t in_total = sig->n_params + (sig->sret ? 1 : 0);
        size_t out_total = (sig->sret || is_void) ? 0 : 1;
        sig->flat_in_tys  = arena_new_array(e->arena, MLIR_TypeHandle, in_total ? in_total : 1);
        sig->flat_out_tys = arena_new_array(e->arena, MLIR_TypeHandle, out_total ? out_total : 1);
        size_t off = 0;
        if (sig->sret) sig->flat_in_tys[off++] = e->ptr;
        for (size_t i = 0; i < sig->n_params; i++) {
            sig->flat_in_tys[off++] = slot_param_type(e, &sig->params[i]);
        }
        sig->n_flat_in = off;
        if (!sig->sret && !is_void) {
            sig->flat_out_tys[0] = scalar_mlir_type(e, sig->ret.type.kind);
            sig->n_flat_out = 1;
        } else {
            sig->n_flat_out = 0;
        }
        sig->fn_ty = MLIR_CreateTypeFunction(e->ctx,
            sig->flat_in_tys, sig->n_flat_in, sig->flat_out_tys, sig->n_flat_out);
        if (sig->is_variadic) {
            // LLVMFunctionType has at most one result; "void" is represented
            // by the LLVM void type rather than a 0-result tuple.
            MLIR_TypeHandle ret_ty = (sig->n_flat_out == 1)
                ? sig->flat_out_tys[0]
                : MLIR_CreateTypeLLVMVoid(e->ctx);
            sig->llvm_fn_ty = MLIR_CreateTypeLLVMFunction(e->ctx,
                ret_ty, sig->flat_in_tys, sig->n_flat_in, /*is_var_arg=*/true);
        }
    }
}

static MLIR_OpHandle emit_func(E *e, Func *f) {
    e->cur_line = f->line;
    FuncSig *sig = find_sig(e, f->name);

    MLIR_RegionHandle body_r = MLIR_CreateRegion(e->ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(e->ctx);
    MLIR_AppendRegionBlock(e->ctx, body_r, entry);

    Scope sc = (Scope){.head = NULL, .parent = NULL};
    MLIR_ValueHandle *flat_args = arena_new_array(e->arena, MLIR_ValueHandle,
                                                  sig->n_flat_in ? sig->n_flat_in : 1);
    for (size_t i = 0; i < sig->n_flat_in; i++) {
        string nm = format(e->arena, str_lit("%arg{}"), (int64_t)i);
        flat_args[i] = MLIR_CreateValueBlockArg(e->ctx, nm, (uint32_t)i,
                                                sig->flat_in_tys[i], eloc(e, 0));
        MLIR_AppendBlockArg(e->ctx, entry, flat_args[i]);
    }

    MLIR_BlockHandle saved = e->cur_block;
    MLIR_RegionHandle saved_region = e->func_region;
    FuncSig *saved_sig = e->cur_sig;
    MLIR_ValueHandle saved_sret = e->cur_sret_ptr;
    e->cur_block = entry;
    e->func_region = body_r;
    e->cur_sig = sig;
    e->cur_sret_ptr = MLIR_INVALID_HANDLE;
    e->terminated = false;
    e->next_ssa = 0;

    size_t arg_off = 0;
    if (sig->sret) {
        e->cur_sret_ptr = flat_args[arg_off++];
    }

    for (size_t i = 0; i < sig->n_params; i++) {
        SlotInfo *p = &sig->params[i];
        Sym *sy = arena_new(e->arena, Sym);
        sy->name = f->params.data[i].name;
        sy->type = p->type;
        MLIR_ValueHandle blk = flat_args[arg_off++];
        if (p->type.kind == TY_STRUCT) {
            // Caller-byval: the block-arg IS our copy. Bind directly.
            sy->sdef = p->sdef;
            sy->addr = blk;
        } else if (p->type.kind == TY_PTR_STRUCT) {
            sy->sdef = p->sdef;
            sy->addr = emit_alloca(e, e->ptr);
            emit_store_v(e, blk, sy->addr);
        } else if (p->type.kind == TY_F32) {
            sy->addr = emit_alloca(e, e->f32);
            emit_store_v(e, blk, sy->addr);
        } else if (p->type.kind == TY_I64) {
            sy->addr = emit_alloca(e, e->i64);
            emit_store_v(e, blk, sy->addr);
        } else if (p->type.kind == TY_PTR_I32 || p->type.kind == TY_PTR_CHAR ||
                   p->type.kind == TY_PTR_VOID || p->type.kind == TY_FNPTR ||
                   p->type.kind == TY_PTR_PTR) {
            sy->addr = emit_alloca(e, e->ptr);
            emit_store_v(e, blk, sy->addr);
        } else {
            sy->addr = emit_alloca(e, e->i32);
            emit_store_v(e, blk, sy->addr);
        }
        sy->next = sc.head;
        sc.head = sy;
    }

    emit_block(e, &sc, f->body);
    if (!e->terminated) {
        if (sig->sret || sig->ret.type.kind == TY_VOID) {
            emit_return_op(e, NULL, 0);
        } else {
            MLIR_TypeHandle ty = sig->flat_out_tys[0];
            MLIR_ValueHandle v = (ty == e->f32) ? emit_const_f32(e, 0.0)
                : (ty == e->i64) ? emit_const_i64(e, 0)
                : (ty == e->ptr) ? emit_null_ptr(e)
                : emit_const_i32(e, 0);
            MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v;
            emit_return_op(e, ops, 1);
        }
        e->terminated = true;
    }
    e->cur_block = saved;
    e->func_region = saved_region;
    e->cur_sig = saved_sig;
    e->cur_sret_ptr = saved_sret;

    MLIR_AttributeHandle sym_name = MLIR_CreateAttributeString(e->ctx, str_lit("sym_name"), f->name);
    // Variadic functions are emitted as `llvm.func` with an LLVMFunctionType
    // attribute (which carries the var_arg flag); non-variadic ones stay as
    // `func.func` with a regular FunctionType.
    MLIR_TypeHandle ft = sig->is_variadic ? sig->llvm_fn_ty : sig->fn_ty;
    MLIR_AttributeHandle fn_ty_attr = MLIR_CreateAttributeType(e->ctx, str_lit("function_type"), ft);
    MLIR_AttributeHandle *attrs = arena_new_array(e->arena, MLIR_AttributeHandle, 2);
    attrs[0] = sym_name; attrs[1] = fn_ty_attr;
    MLIR_RegionHandle *regs = arena_new_array(e->arena, MLIR_RegionHandle, 1); regs[0] = body_r;
    MLIR_OpHandle fn;
    if (sig->is_variadic) {
        fn = MLIR_CreateOp(e->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.func"),
                           attrs, 2, NULL, 0, NULL, 0, NULL, 0,
                           regs, 1, eloc(e, 0), MLIR_INVALID_HANDLE,
                           str_lit(""), -1);
    } else {
        fn = MLIR_CreateOp(e->ctx, OP_TYPE_FUNC_FUNC, str_lit("func.func"),
                           attrs, 2, NULL, 0, NULL, 0, NULL, 0,
                           regs, 1, eloc(e, 0), MLIR_INVALID_HANDLE,
                           str_lit(""), -1);
    }
    return fn;
}

// Compute byte size per a fixed layout: i32/f32 = 4, ptr = 8, struct =
// padded sum of fields (round each field to its alignment, struct align =
// max field align, total rounded up to that). Used for sizeof().
static int64_t type_align(E *e, Type t);

// Infer the static C-level Type of an expression for `sizeof(<expr>)`.
// We do NOT emit any operations — we walk the AST and look up symbols /
// struct fields by name. Returns TY_VOID to signal "unsupported" (the
// caller emits an error). Side-effecting expressions are still safe
// because we never evaluate them.
static Type infer_expr_type(E *e, Scope *sc, Expr *ex) {
    Type t = (Type){0};
    switch (ex->kind) {
        case EX_INT:
            if (ex->is_i64) { t.kind = TY_I64; t.int_bits = 64; }
            else            { t.kind = TY_I32; t.int_bits = 32; }
            return t;
        case EX_FLOAT: t.kind = TY_F32; return t;
        case EX_STR:   t.kind = TY_PTR_CHAR; return t;
        case EX_NULL:  t.kind = TY_PTR_VOID; return t;
        case EX_VAR: {
            Sym *s = lookup(e, sc, ex->name);
            if (s) return s->type;
            int64_t en;
            if (find_enum(e, ex->name, &en)) { t.kind = TY_I32; return t; }
            return t;
        }
        case EX_DEREF: {
            Type inner = infer_expr_type(e, sc, ex->lhs);
            if (inner.kind == TY_PTR_I32) { t.kind = TY_I32; return t; }
            if (inner.kind == TY_PTR_CHAR) { t.kind = TY_I32; return t; }
            if (inner.kind == TY_PTR_PTR && inner.pointee) return *inner.pointee;
            if (inner.kind == TY_PTR_STRUCT) {
                t.kind = TY_STRUCT; t.struct_name = inner.struct_name; return t;
            }
            return t;
        }
        case EX_ADDR: {
            Type inner = infer_expr_type(e, sc, ex->lhs);
            if (inner.kind == TY_I32) { t.kind = TY_PTR_I32; return t; }
            if (inner.kind == TY_STRUCT) {
                t.kind = TY_PTR_STRUCT; t.struct_name = inner.struct_name;
                return t;
            }
            // For other base kinds we report a generic pointer; sizeof
            // is the same regardless.
            t.kind = TY_PTR_VOID; return t;
        }
        case EX_FIELD: {
            Type st = infer_expr_type(e, sc, ex->lhs);
            StructDef *sd = NULL;
            if (st.kind == TY_STRUCT)      sd = find_struct(e, st.struct_name);
            else if (st.kind == TY_PTR_STRUCT) sd = find_struct(e, st.struct_name);
            if (!sd) return t;
            for (size_t i = 0; i < sd->fields.size; i++) {
                if (str_eq(sd->fields.data[i].name, ex->name))
                    return sd->fields.data[i].type;
            }
            return t;
        }
        case EX_INDEX: {
            Type base = infer_expr_type(e, sc, ex->lhs);
            if (base.kind == TY_ARRAY_I32 || base.kind == TY_PTR_I32) {
                t.kind = TY_I32; return t;
            }
            if (base.kind == TY_PTR_CHAR) { t.kind = TY_I32; return t; }
            if (base.kind == TY_ARRAY_F32) { t.kind = TY_F32; return t; }
            if (base.kind == TY_ARRAY_STRUCT) {
                t.kind = TY_STRUCT; t.struct_name = base.struct_name; return t;
            }
            return t;
        }
        case EX_CALL: {
            FuncSig *sig = find_sig(e, ex->callee);
            if (sig) return sig->ret.type;
            return t;
        }
        case EX_CAST: return ex->cast_type;
        case EX_COMPOUND: return ex->cast_type;
        case EX_SIZEOF: t.kind = TY_I32; return t;
        case EX_BIN: {
            Type a = infer_expr_type(e, sc, ex->lhs);
            Type b = infer_expr_type(e, sc, ex->rhs);
            // Usual arithmetic conversions (simplified): if either side
            // is i64 the result is i64; if either is f32, result is f32.
            if (a.kind == TY_F32 || b.kind == TY_F32) { t.kind = TY_F32; return t; }
            if (a.kind == TY_I64 || b.kind == TY_I64) { t.kind = TY_I64; return t; }
            return a;
        }
        case EX_UN:
        case EX_TERNARY:
        case EX_ASSIGN: {
            // Use the LHS type as the result type.
            return infer_expr_type(e, sc, ex->lhs ? ex->lhs : ex->lvalue);
        }
        default: return t;
    }
}

static int64_t type_size(E *e, Type t) {
    if (t.kind == TY_I32) return 4;
    if (t.kind == TY_I64) return 8;
    if (t.kind == TY_F32) return 4;
    if (t.kind == TY_PTR_I32 || t.kind == TY_PTR_STRUCT) return 8;
    if (t.kind == TY_PTR_CHAR || t.kind == TY_PTR_VOID || t.kind == TY_FNPTR) return 8;
    if (t.kind == TY_PTR_PTR) return 8;
    if (t.kind == TY_ARRAY_I32) return 4 * t.array_len * (t.array_len2 ? t.array_len2 : 1);
    if (t.kind == TY_ARRAY_F32) return 4 * t.array_len * (t.array_len2 ? t.array_len2 : 1);
    if (t.kind == TY_ARRAY_PTR_STRUCT || t.kind == TY_ARRAY_PTR_CHAR)
        return 8 * t.array_len;
    if (t.kind == TY_STRUCT) {
        StructDef *sd = find_struct(e, t.struct_name);
        if (!sd) return 0;
        int64_t off = 0, max_align = 1;
        for (size_t i = 0; i < sd->fields.size; i++) {
            Type ft = sd->fields.data[i].type;
            int64_t fa = type_align(e, ft);
            int64_t fs = type_size(e, ft);
            if (fa > max_align) max_align = fa;
            // round off up to fa
            off = (off + fa - 1) / fa * fa;
            off += fs;
        }
        if (max_align > 0) off = (off + max_align - 1) / max_align * max_align;
        return off;
    }
    if (t.kind == TY_ARRAY_STRUCT) {
        Type elt = (Type){.kind = TY_STRUCT, .struct_name = t.struct_name};
        return type_size(e, elt) * t.array_len;
    }
    return 0;
}
static int64_t type_align(E *e, Type t) {
    if (t.kind == TY_I32 || t.kind == TY_F32) return 4;
    if (t.kind == TY_I64) return 8;
    if (t.kind == TY_PTR_I32 || t.kind == TY_PTR_STRUCT) return 8;
    if (t.kind == TY_PTR_CHAR || t.kind == TY_PTR_VOID || t.kind == TY_FNPTR) return 8;
    if (t.kind == TY_PTR_PTR) return 8;
    if (t.kind == TY_ARRAY_I32 || t.kind == TY_ARRAY_F32) return 4;
    if (t.kind == TY_ARRAY_PTR_STRUCT || t.kind == TY_ARRAY_PTR_CHAR) return 8;
    if (t.kind == TY_STRUCT) {
        StructDef *sd = find_struct(e, t.struct_name);
        if (!sd) return 1;
        int64_t a = 1;
        for (size_t i = 0; i < sd->fields.size; i++) {
            int64_t fa = type_align(e, sd->fields.data[i].type);
            if (fa > a) a = fa;
        }
        return a;
    }
    if (t.kind == TY_ARRAY_STRUCT) {
        Type elt = (Type){.kind = TY_STRUCT, .struct_name = t.struct_name};
        return type_align(e, elt);
    }
    return 1;
}

// Detect by-value cycles in struct definitions (TY_STRUCT field edges
// only; pointer fields don't count). Standard 3-color DFS.
static bool struct_cycle_dfs(E *e, StructDef *sd, int *color, StructDef **stack, int depth) {
    e->cur_line = sd->line;
    size_t idx = 0;
    for (; idx < e->n_struct_types && e->struct_types[idx].sd != sd; idx++) {}
    if (idx >= e->n_struct_types) return false;
    if (color[idx] == 1) {
        EMIT_ERR(e, "by-value cyclic struct definition involving '{}'",
                sd->name);
        return true;
    }
    if (color[idx] == 2) return false;
    color[idx] = 1;
    for (size_t i = 0; i < sd->fields.size; i++) {
        Type ft = sd->fields.data[i].type;
        if (ft.kind == TY_STRUCT) {
            StructDef *inner = find_struct(e, ft.struct_name);
            if (inner && struct_cycle_dfs(e, inner, color, stack, depth + 1)) return true;
        }
    }
    color[idx] = 2;
    return false;
}

static void check_struct_cycles(E *e) {
    if (e->n_struct_types == 0) return;
    int *color = arena_new_array(e->arena, int, e->n_struct_types);
    for (size_t i = 0; i < e->n_struct_types; i++) color[i] = 0;
    StructDef **stack = arena_new_array(e->arena, StructDef *, e->n_struct_types);
    for (size_t i = 0; i < e->n_struct_types; i++) {
        if (color[i] == 0) {
            e->cur_line = e->struct_types[i].sd->line;
            if (struct_cycle_dfs(e, e->struct_types[i].sd, color, stack, 0)) {
                EMIT_ERR(e, "aborting due to cyclic struct definition");
                exit(1);
            }
        }
    }
}

// Pre-pass: register identified !llvm.struct types for each StructDef and
// then set their bodies (allowing already-registered types to be looked up
// when nested by-value structs reference each other).
static void init_struct_types(E *e) {
    e->n_struct_types = e->program->structs.size;
    e->struct_types = arena_new_array(e->arena, StructTypeEntry,
                                      e->n_struct_types ? e->n_struct_types : 1);
    for (size_t i = 0; i < e->n_struct_types; i++) {
        StructDef *sd = e->program->structs.data[i];
        e->struct_types[i].sd = sd;
        e->struct_types[i].ty = MLIR_CreateTypeLLVMStructIdentified(e->ctx, sd->name);
    }
    for (size_t i = 0; i < e->n_struct_types; i++) {
        StructDef *sd = e->struct_types[i].sd;
        e->cur_line = sd->line;
        size_t n = sd->fields.size;
        MLIR_TypeHandle *body = arena_new_array(e->arena, MLIR_TypeHandle, n ? n : 1);
        for (size_t k = 0; k < n; k++) {
            Type ft = sd->fields.data[k].type;
            if (ft.kind == TY_I32) body[k] = e->i32;
            else if (ft.kind == TY_I64) body[k] = e->i64;
            else if (ft.kind == TY_F32) body[k] = e->f32;
            else if (ft.kind == TY_PTR_STRUCT || ft.kind == TY_PTR_I32 ||
                     ft.kind == TY_PTR_CHAR || ft.kind == TY_PTR_VOID) body[k] = e->ptr;
            else if (ft.kind == TY_STRUCT) {
                StructDef *inner = find_struct(e, ft.struct_name);
                MLIR_TypeHandle t = find_struct_type(e, inner);
                body[k] = t;
            } else if (ft.kind == TY_ARRAY_I32 || ft.kind == TY_ARRAY_F32) {
                MLIR_TypeHandle elem = (ft.kind == TY_ARRAY_F32) ? e->f32 : e->i32;
                MLIR_TypeHandle inner;
                if (ft.array_len2 != 0) {
                    MLIR_TypeHandle in2 = MLIR_CreateTypeLLVMArray(
                        e->ctx, elem, (uint64_t)ft.array_len2);
                    inner = MLIR_CreateTypeLLVMArray(
                        e->ctx, in2, (uint64_t)ft.array_len);
                } else {
                    inner = MLIR_CreateTypeLLVMArray(
                        e->ctx, elem, (uint64_t)ft.array_len);
                }
                body[k] = inner;
            } else if (ft.kind == TY_ARRAY_PTR_STRUCT ||
                       ft.kind == TY_ARRAY_PTR_CHAR) {
                body[k] = MLIR_CreateTypeLLVMArray(
                    e->ctx, e->ptr, (uint64_t)ft.array_len);
            } else {
                EMIT_ERR(e, "unsupported struct field type in {}", sd->name);
                body[k] = e->i32;
            }
        }
        MLIR_SetTypeLLVMStructBody(e->ctx, e->struct_types[i].ty, body, n);
    }
}

MLIR_OpHandle tinyc_emit_module(MLIR_Context *ctx, Program *program) {
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    E e = (E){0};
    e.ctx = ctx;
    e.arena = arena;
    e.i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    e.i1  = MLIR_CreateTypeInteger(ctx, 1, false);
    e.i64 = MLIR_CreateTypeInteger(ctx, 64, true);
    e.i8  = MLIR_CreateTypeInteger(ctx, 8, true);
    e.f32 = MLIR_CreateTypeFloat(ctx, 32, false);
    e.index = MLIR_CreateTypeIndex(ctx);
    e.ptr = MLIR_CreateTypeLLVMPointer(ctx);
    e.loc = MLIR_CreateLocationUnknown(ctx, str_lit(""));
    e.next_ssa = 0;
    e.cur_block = MLIR_INVALID_HANDLE;
    e.func_region = MLIR_INVALID_HANDLE;
    e.terminated = false;
    e.loops = NULL;
    e.program = program;
    e.cur_sig = NULL;
    e.cur_sret_ptr = MLIR_INVALID_HANDLE;

    init_struct_types(&e);
    check_struct_cycles(&e);
    build_signatures(&e);

    MLIR_RegionHandle mr = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle mb = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, mr, mb);
    e.module_block = mb;
    MLIR_RegionHandle *mregs = arena_new_array(arena, MLIR_RegionHandle, 1); mregs[0] = mr;
    MLIR_OpHandle module = MLIR_CreateOp(ctx, OP_TYPE_MODULE, str_lit("module"),
                                         NULL, 0, NULL, 0, NULL, 0, NULL, 0,
                                         mregs, 1, e.loc, MLIR_INVALID_HANDLE,
                                         str_lit(""), -1);

    // Pre-pass: emit module-level globals. For char* globals initialized
    // by a string literal, emit the string-literal global first, then the
    // pointer global with an initializer region that takes the address.
    for (size_t i = 0; i < program->globals.size; i++) {
        Global *g = &program->globals.data[i];
        Sym *sy = arena_new(arena, Sym);
        *sy = (Sym){0};
        sy->name = g->name;
        sy->type = g->type;
        sy->is_global = true;
        sy->global_sym = g->name;
        sy->next = e.globals;
        e.globals = sy;

        if (g->is_extern) {
            // `extern T x;` — emit external linkage with no initializer.
            // Used for libc symbols like `stdout` resolved by the linker.
            MLIR_TypeHandle gty;
            if (g->type.kind == TY_F32) {
                gty = e.f32;
            } else if (g->type.kind == TY_PTR_CHAR || g->type.kind == TY_PTR_I32 ||
                       g->type.kind == TY_PTR_STRUCT || g->type.kind == TY_PTR_VOID ||
                       g->type.kind == TY_FNPTR) {
                gty = e.ptr;
            } else {
                gty = e.i32;
            }
            MLIR_OpHandle gop = MLIR_CreateLLVMGlobal(ctx, g->name, gty,
                /*is_constant=*/false,
                /*init_kind=*/4, 0, 0.0, NULL, e.loc);
            MLIR_AppendBlockOp(ctx, mb, gop);
        } else if (g->type.kind == TY_PTR_CHAR) {
            string str_sym = (string){0};
            if (g->has_init && g->init_str.size > 0) {
                str_sym = intern_string(&e, g->init_str);
            }
            // Emit the !llvm.ptr global with an initializer region that
            // either returns the string addr or a null pointer.
            MLIR_BlockHandle init_blk = MLIR_INVALID_HANDLE;
            MLIR_OpHandle gop = MLIR_CreateLLVMGlobal(ctx, g->name, e.ptr,
                /*is_constant=*/false,
                /*init_kind=*/2, 0, 0.0, &init_blk, e.loc);
            // Build initializer body: %0 = addressof @str ; llvm.return %0
            MLIR_BlockHandle save_blk = e.cur_block;
            bool save_term = e.terminated;
            e.cur_block = init_blk;
            e.terminated = false;
            MLIR_ValueHandle init_v;
            if (str_sym.size > 0) {
                init_v = emit_addressof(&e, str_sym);
            } else {
                init_v = emit_null_ptr(&e);
            }
            MLIR_ValueHandle *rops = arena_new_array(arena, MLIR_ValueHandle, 1);
            rops[0] = init_v;
            emit_op(&e, OP_TYPE_LLVM_RETURN, str_lit("llvm.return"),
                    NULL, 0, NULL, 0, rops, 1, NULL, 0, NULL, 0);
            e.cur_block = save_blk;
            e.terminated = save_term;
            MLIR_AppendBlockOp(ctx, mb, gop);
        } else if (g->type.kind == TY_F32) {
            MLIR_OpHandle gop = MLIR_CreateLLVMGlobal(ctx, g->name, e.f32,
                /*is_constant=*/false,
                /*init_kind=*/1, 0, g->has_init ? g->init_float : 0.0,
                NULL, e.loc);
            MLIR_AppendBlockOp(ctx, mb, gop);
        } else if (g->type.kind == TY_PTR_I32 || g->type.kind == TY_PTR_STRUCT ||
                   g->type.kind == TY_PTR_VOID || g->type.kind == TY_FNPTR) {
            // Emit a zero-initialized pointer global.
            MLIR_BlockHandle init_blk = MLIR_INVALID_HANDLE;
            MLIR_OpHandle gop = MLIR_CreateLLVMGlobal(ctx, g->name, e.ptr,
                /*is_constant=*/false,
                /*init_kind=*/2, 0, 0.0, &init_blk, e.loc);
            MLIR_BlockHandle save_blk = e.cur_block;
            bool save_term = e.terminated;
            e.cur_block = init_blk;
            e.terminated = false;
            MLIR_ValueHandle init_v = emit_null_ptr(&e);
            MLIR_ValueHandle *rops = arena_new_array(arena, MLIR_ValueHandle, 1);
            rops[0] = init_v;
            emit_op(&e, OP_TYPE_LLVM_RETURN, str_lit("llvm.return"),
                    NULL, 0, NULL, 0, rops, 1, NULL, 0, NULL, 0);
            e.cur_block = save_blk;
            e.terminated = save_term;
            MLIR_AppendBlockOp(ctx, mb, gop);
        } else {
            // i32 (default for plain int)
            MLIR_OpHandle gop = MLIR_CreateLLVMGlobal(ctx, g->name, e.i32,
                /*is_constant=*/false,
                /*init_kind=*/0, g->has_init ? g->init_int : 0, 0.0,
                NULL, e.loc);
            MLIR_AppendBlockOp(ctx, mb, gop);
        }
    }

    for (size_t i = 0; i < program->funcs.size; i++) {
        if (program->funcs.data[i]->is_forward) continue;
        MLIR_OpHandle fn = emit_func(&e, program->funcs.data[i]);
        MLIR_AppendBlockOp(ctx, mb, fn);
    }

    // Emit extern declarations for any forward-declared functions that
    // never received a definition. Without these, `func.call` / `llvm.call`
    // ops referring to them would fail symbol verification. Variadic decls
    // become `llvm.func` (so the variadic flag is preserved); non-variadic
    // ones use `func.func` matching the existing malloc/free pattern.
    for (size_t i = 0; i < program->funcs.size; i++) {
        Func *fwd = program->funcs.data[i];
        if (!fwd->is_forward) continue;
        FuncSig *sig = NULL;
        for (size_t j = 0; j < e.n_sigs; j++) {
            if (str_eq(e.sigs[j].name, fwd->name)) { sig = &e.sigs[j]; break; }
        }
        if (!sig) continue;
        MLIR_AttributeHandle a0 = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), fwd->name);
        MLIR_TypeHandle ft = sig->is_variadic ? sig->llvm_fn_ty : sig->fn_ty;
        MLIR_AttributeHandle a1 = MLIR_CreateAttributeType(ctx, str_lit("function_type"), ft);
        MLIR_AttributeHandle a2 = MLIR_CreateAttributeString(ctx, str_lit("sym_visibility"), str_lit("private"));
        MLIR_AttributeHandle *attrs = arena_new_array(arena, MLIR_AttributeHandle, 3);
        attrs[0] = a0; attrs[1] = a1; attrs[2] = a2;
        MLIR_RegionHandle body = MLIR_CreateRegion(ctx);
        MLIR_RegionHandle *regs = arena_new_array(arena, MLIR_RegionHandle, 1); regs[0] = body;
        MLIR_OpHandle decl;
        if (sig->is_variadic) {
            decl = MLIR_CreateOp(ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.func"),
                                 attrs, 3, NULL, 0, NULL, 0, NULL, 0,
                                 regs, 1, e.loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
        } else {
            decl = MLIR_CreateOp(ctx, OP_TYPE_FUNC_FUNC, str_lit("func.func"),
                                 attrs, 3, NULL, 0, NULL, 0, NULL, 0,
                                 regs, 1, e.loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
        }
        MLIR_AppendBlockOp(ctx, mb, decl);
    }

    // Emit `malloc`/`free` extern declarations at module scope so that
    // user code calling them links against libc — unless the user has
    // already provided their own forward declaration above (which would
    // otherwise create duplicate symbols and fail symbol verification).
    bool have_user_malloc = false, have_user_free = false;
    for (size_t i = 0; i < program->funcs.size; i++) {
        Func *fwd = program->funcs.data[i];
        if (!fwd->is_forward) continue;
        if (str_eq(fwd->name, str_lit("malloc"))) have_user_malloc = true;
        else if (str_eq(fwd->name, str_lit("free"))) have_user_free = true;
    }
    if (!have_user_malloc) {
        MLIR_TypeHandle *ins = arena_new_array(arena, MLIR_TypeHandle, 1); ins[0] = e.i64;
        MLIR_TypeHandle *outs = arena_new_array(arena, MLIR_TypeHandle, 1); outs[0] = e.ptr;
        MLIR_TypeHandle fty = MLIR_CreateTypeFunction(ctx, ins, 1, outs, 1);
        MLIR_AttributeHandle a0 = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), str_lit("malloc"));
        MLIR_AttributeHandle a1 = MLIR_CreateAttributeType(ctx, str_lit("function_type"), fty);
        MLIR_AttributeHandle a2 = MLIR_CreateAttributeString(ctx, str_lit("sym_visibility"), str_lit("private"));
        MLIR_AttributeHandle *attrs = arena_new_array(arena, MLIR_AttributeHandle, 3);
        attrs[0] = a0; attrs[1] = a1; attrs[2] = a2;
        MLIR_RegionHandle body = MLIR_CreateRegion(ctx);
        MLIR_RegionHandle *regs = arena_new_array(arena, MLIR_RegionHandle, 1); regs[0] = body;
        MLIR_OpHandle decl = MLIR_CreateOp(ctx, OP_TYPE_FUNC_FUNC, str_lit("func.func"),
                                           attrs, 3, NULL, 0, NULL, 0, NULL, 0,
                                           regs, 1, e.loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
        MLIR_AppendBlockOp(ctx, mb, decl);
    }
    if (!have_user_free) {
        MLIR_TypeHandle *ins = arena_new_array(arena, MLIR_TypeHandle, 1); ins[0] = e.ptr;
        MLIR_TypeHandle fty = MLIR_CreateTypeFunction(ctx, ins, 1, NULL, 0);
        MLIR_AttributeHandle a0 = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), str_lit("free"));
        MLIR_AttributeHandle a1 = MLIR_CreateAttributeType(ctx, str_lit("function_type"), fty);
        MLIR_AttributeHandle a2 = MLIR_CreateAttributeString(ctx, str_lit("sym_visibility"), str_lit("private"));
        MLIR_AttributeHandle *attrs = arena_new_array(arena, MLIR_AttributeHandle, 3);
        attrs[0] = a0; attrs[1] = a1; attrs[2] = a2;
        MLIR_RegionHandle body = MLIR_CreateRegion(ctx);
        MLIR_RegionHandle *regs = arena_new_array(arena, MLIR_RegionHandle, 1); regs[0] = body;
        MLIR_OpHandle decl = MLIR_CreateOp(ctx, OP_TYPE_FUNC_FUNC, str_lit("func.func"),
                                           attrs, 3, NULL, 0, NULL, 0, NULL, 0,
                                           regs, 1, e.loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
        MLIR_AppendBlockOp(ctx, mb, decl);
    }

    // tinyc_va_arg_* helpers (defined in runtime.c) — emitted on demand
    // when the user code uses va_arg.
    if (e.need_va_arg_helpers) {
        struct { string name; MLIR_TypeHandle ret; } helpers[] = {
            { str_lit("tinyc_va_arg_i32"), e.i32 },
            { str_lit("tinyc_va_arg_i64"), e.i64 },
            { str_lit("tinyc_va_arg_ptr"), e.ptr },
        };
        for (size_t k = 0; k < 3; k++) {
            MLIR_TypeHandle *ins = arena_new_array(arena, MLIR_TypeHandle, 1); ins[0] = e.ptr;
            MLIR_TypeHandle *outs = arena_new_array(arena, MLIR_TypeHandle, 1); outs[0] = helpers[k].ret;
            MLIR_TypeHandle fty = MLIR_CreateTypeFunction(ctx, ins, 1, outs, 1);
            MLIR_AttributeHandle a0 = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), helpers[k].name);
            MLIR_AttributeHandle a1 = MLIR_CreateAttributeType(ctx, str_lit("function_type"), fty);
            MLIR_AttributeHandle a2 = MLIR_CreateAttributeString(ctx, str_lit("sym_visibility"), str_lit("private"));
            MLIR_AttributeHandle *attrs = arena_new_array(arena, MLIR_AttributeHandle, 3);
            attrs[0] = a0; attrs[1] = a1; attrs[2] = a2;
            MLIR_RegionHandle body = MLIR_CreateRegion(ctx);
            MLIR_RegionHandle *regs = arena_new_array(arena, MLIR_RegionHandle, 1); regs[0] = body;
            MLIR_OpHandle decl = MLIR_CreateOp(ctx, OP_TYPE_FUNC_FUNC, str_lit("func.func"),
                                               attrs, 3, NULL, 0, NULL, 0, NULL, 0,
                                               regs, 1, e.loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
            MLIR_AppendBlockOp(ctx, mb, decl);
        }
    }

    // Emit any string-literal globals collected by ST_PRINT/EX_STR.
    for (size_t i = 0; i < e.n_strings; i++) {
        StringEntry *se = &e.strings[i];
        MLIR_OpHandle gop = MLIR_CreateLLVMGlobalString(
            ctx, se->sym, se->bytes, e.loc);
        MLIR_AppendBlockOp(ctx, mb, gop);
    }

    // If any user-code path used print(<string>), declare @printStr.
    if (e.use_print_str) {
        MLIR_TypeHandle *ins = arena_new_array(arena, MLIR_TypeHandle, 1); ins[0] = e.ptr;
        MLIR_TypeHandle fty = MLIR_CreateTypeFunction(ctx, ins, 1, NULL, 0);
        MLIR_AttributeHandle a0 = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), str_lit("printStr"));
        MLIR_AttributeHandle a1 = MLIR_CreateAttributeType(ctx, str_lit("function_type"), fty);
        MLIR_AttributeHandle a2 = MLIR_CreateAttributeString(ctx, str_lit("sym_visibility"), str_lit("private"));
        MLIR_AttributeHandle *attrs = arena_new_array(arena, MLIR_AttributeHandle, 3);
        attrs[0] = a0; attrs[1] = a1; attrs[2] = a2;
        MLIR_RegionHandle body = MLIR_CreateRegion(ctx);
        MLIR_RegionHandle *regs = arena_new_array(arena, MLIR_RegionHandle, 1); regs[0] = body;
        MLIR_OpHandle decl = MLIR_CreateOp(ctx, OP_TYPE_FUNC_FUNC, str_lit("func.func"),
                                           attrs, 3, NULL, 0, NULL, 0, NULL, 0,
                                           regs, 1, e.loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
        MLIR_AppendBlockOp(ctx, mb, decl);
    }
    return module;
}

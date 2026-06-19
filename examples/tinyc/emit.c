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
    bool           is_used;          // any reachable call site referenced this
                                     // signature; used to suppress extern
                                     // decls for forward-declared functions
                                     // that are never called (otherwise they
                                     // turn into spurious wasm imports that
                                     // fail to resolve at link time).
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
    MLIR_TypeHandle f64;
    MLIR_TypeHandle index;
    MLIR_TypeHandle ptr;             // !llvm.ptr
    MLIR_TypeHandle size_t_ty;       // i32 on wasm32, i64 on native (matches `unsigned long`)
    bool             target_wasm32;  // mirrors program->target_wasm32, cached on E for convenience
    MLIR_BlockHandle cur_block;
    MLIR_BlockHandle entry_block;    // function entry block, alloca insertion point
    MLIR_ValueHandle entry_const_one; // i64 constant 1 in entry_block, shared by all allocas
    size_t entry_alloca_insert_idx;  // next position in entry_block for hoisted allocas
    bool terminated;
    MLIR_RegionHandle func_region;
    MLIR_LocationHandle loc;
    int next_ssa;
    LoopCtx *loops;
    // Linked list of `goto` labels in the currently-emitting function.
    // Pre-populated at emit_func entry with one entry per ST_LABEL found
    // anywhere in the body. Reset to NULL between functions.
    struct LabelBlock {
        string name;
        MLIR_BlockHandle block;
        struct LabelBlock *next;
    } *labels;
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
    bool             need_printf_decl; // emit @printf extern decl when absent
    bool             use_print_str;    // wasm path: emit @printStr extern decl
    bool             need_va_arg_helpers;  // emit tinyc_va_arg_* externs
    bool             need_va_arg_struct;   // emit tinyc_va_arg_struct extern
    bool             need_syscall6_stub;   // emit @__tinyc_syscall6 extern decl
    int              cur_line;       // last AST node line entered; used by
                                     // EMIT_ERR for diagnostic line numbers.
    int              err_count;      // count of EMIT_ERR diagnostics
} E;

// All emit-time diagnostics route through this macro so they carry a line
// number. `msg` must be a C string literal (it is concatenated with the
// "tinyc emit error at line {}: " prefix at preprocess time and wrapped in
// str_lit). The line is taken from e->cur_line, which is updated at the
// entry of emit_expr / emit_lvalue / emit_stmt / emit_func / per-struct
// pre-pass loops.
#define EMIT_ERR(e, msg, ...) \
    do { \
        println(str_lit("tinyc emit error at line {}: " msg), \
                (int64_t)((e)->cur_line), ##__VA_ARGS__); \
        (e)->err_count++; \
    } while (0)

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
        if (str_eq(e->sigs[i].name, name)) {
            // Mark the signature as referenced. Forward-declared
            // functions that never end up being looked up here are
            // skipped in the late `emit extern decls` pass — emitting
            // a `func.func` declaration for an unreferenced extern
            // would otherwise turn into a spurious wasm import that
            // fails to resolve at link time.
            e->sigs[i].is_used = true;
            return &e->sigs[i];
        }
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
    if (k == TY_F64) return e->f64;
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
    // Append the new block to the region that currently owns
    // `e->cur_block`, not blindly to the function body. The CFG-based
    // emitters (if, while, ternary, ...) can run while emit_expr is
    // recursing inside another op's region (e.g. an scf.if then-block
    // emitted by &&/||): the ternary's new blocks must live in that
    // inner region, otherwise the resulting IR has cf.cond_br ops
    // pointing at successors outside their own region and downstream
    // passes (cf->scf lift, wasmssa lower, ...) reject it.
    MLIR_RegionHandle target = e->func_region;
    if (e->cur_block != MLIR_INVALID_HANDLE) {
        MLIR_RegionHandle r = MLIR_GetBlockParentRegion(e->cur_block);
        if (r != MLIR_INVALID_HANDLE) target = r;
    }
    MLIR_AppendRegionBlock(e->ctx, target, b);
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

static MLIR_ValueHandle emit_const_f64(E *e, double v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->f64, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->f64;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_AttributeHandle val = MLIR_CreateAttributeFloat(e->ctx, str_lit("value"), v, e->f64);
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = val;
    emit_op(e, OP_TYPE_ARITH_CONSTANT, str_lit("arith.constant"),
            rts, 1, rs, 1, NULL, 0, as, 1, NULL, 0);
    return r;
}

// arith.extf : f32 -> f64
static MLIR_ValueHandle emit_fpext_f32_to_f64(E *e, MLIR_ValueHandle v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->f64, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->f64;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v;
    emit_op(e, OP_TYPE_UNREGISTERED, str_lit("arith.extf"),
            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
    return r;
}

// arith.truncf : f64 -> f32
static MLIR_ValueHandle emit_fptrunc_f64_to_f32(E *e, MLIR_ValueHandle v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->f32, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->f32;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v;
    emit_op(e, OP_TYPE_UNREGISTERED, str_lit("arith.truncf"),
            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
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

// Zero-extend i32 -> i64 (arith.extui), used to widen unsigned operands.
static MLIR_ValueHandle emit_extui_i32_to_i64(E *e, MLIR_ValueHandle v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i64, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i64;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v;
    emit_op(e, OP_TYPE_ARITH_EXTUI, str_lit("arith.extui"),
            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
    return r;
}

// Reinterpret a pointer as an i64 (llvm.ptrtoint). Used to pass pointer
// arguments to __builtin_syscall6 without requiring an explicit (long) cast.
static MLIR_ValueHandle emit_ptrtoint_i64(E *e, MLIR_ValueHandle v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i64, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i64;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v;
    emit_op(e, OP_TYPE_LLVM_PTRTOINT, str_lit("llvm.ptrtoint"),
            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
    return r;
}

// Sign-extend i8 -> i32 (arith.extsi).
static MLIR_ValueHandle emit_extsi_i8_to_i32(E *e, MLIR_ValueHandle v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i32, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i32;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v;
    emit_op(e, OP_TYPE_ARITH_EXTSI, str_lit("arith.extsi"),
            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
    return r;
}

// Zero-extend i8 -> i32 (arith.extui), used to widen loads from
// `unsigned char` / `uint8_t` arrays and pointers.
static MLIR_ValueHandle emit_extui_i8_to_i32(E *e, MLIR_ValueHandle v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i32, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i32;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v;
    emit_op(e, OP_TYPE_ARITH_EXTUI, str_lit("arith.extui"),
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

// Truncate any integer -> i8 (arith.trunci).
static MLIR_ValueHandle emit_trunci_to_i8(E *e, MLIR_ValueHandle v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i8, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i8;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v;
    emit_op(e, OP_TYPE_ARITH_TRUNCI, str_lit("arith.trunci"),
            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
    return r;
}

// arith.constant <v> : i8 — used for char-array string-literal init.
static MLIR_ValueHandle emit_const_i8(E *e, int64_t v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i8, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i8;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_AttributeHandle val = MLIR_CreateAttributeInteger(e->ctx, str_lit("value"), v, e->i8);
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = val;
    emit_op(e, OP_TYPE_ARITH_CONSTANT, str_lit("arith.constant"),
            rts, 1, rs, 1, NULL, 0, as, 1, NULL, 0);
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

static MLIR_ValueHandle emit_string_ptr(E *e, string bytes) {
    return emit_addressof(e, intern_string(e, bytes));
}

static void emit_printf_call(E *e, MLIR_ValueHandle fmt, MLIR_ValueHandle arg,
                             bool has_arg) {
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, has_arg ? 2 : 1);
    ops[0] = fmt;
    if (has_arg) ops[1] = arg;
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1);
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1);
    rts[0] = e->i32;
    rs[0] = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                     e->i32, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle ins[1] = { e->ptr };
    MLIR_TypeHandle fn_ty = MLIR_CreateTypeLLVMFunction(e->ctx, e->i32, ins, 1, true);
    MLIR_AttributeHandle ca = MLIR_CreateAttributeSymbolRef(
        e->ctx, str_lit("callee"), str_lit("printf"));
    MLIR_AttributeHandle vt = MLIR_CreateAttributeType(
        e->ctx, str_lit("var_callee_type"), fn_ty);
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 2);
    as[0] = ca; as[1] = vt;
    emit_op(e, OP_TYPE_UNREGISTERED, str_lit("llvm.call"),
            rts, 1, rs, 1, ops, has_arg ? 2 : 1, as, 2, NULL, 0);
    e->need_printf_decl = true;
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
    // Hoist all `llvm.alloca` ops to the function's entry block so loops
    // (where local declarations sit on each iteration) don't grow the stack
    // unboundedly. We share a single `arith.constant 1 : i64` count operand
    // that lives at the top of the entry block.
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->ptr, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->ptr;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1);
    ops[0] = e->entry_const_one;
    MLIR_AttributeHandle elem_attr = MLIR_CreateAttributeType(e->ctx, str_lit("elem_type"), elem_ty);
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = elem_attr;

    bool hoist = e->entry_block && e->entry_block != e->cur_block;
    MLIR_BlockHandle saved = e->cur_block;
    if (hoist) e->cur_block = MLIR_INVALID_HANDLE;
    MLIR_OpHandle aop = emit_op(e, OP_TYPE_LLVM_ALLOCA, str_lit("llvm.alloca"),
                                rts, 1, rs, 1, ops, 1, as, 1, NULL, 0);
    if (hoist) {
        // Insert at a stable position right after entry_const_one (and any
        // previously-hoisted allocas) so the alloca dominates *all* uses,
        // including ones nested inside scf.if / cf.cond_br regions whose
        // ops were already appended to the entry block.
        MLIR_InsertBlockOpAtIndex(e->ctx, e->entry_block, aop,
                                  e->entry_alloca_insert_idx++);
        e->cur_block = saved;
    } else if (e->cur_block == e->entry_block) {
        // Alloca emitted directly into entry block: track that it sits
        // before any later body ops so future hoisted allocas are inserted
        // after it.
        e->entry_alloca_insert_idx++;
    }
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

// arith.sitofp : iN -> f64. iN may be i32 or i64.
static MLIR_ValueHandle emit_sitofp_f64(E *e, MLIR_ValueHandle iv) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->f64, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->f64;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = iv;
    emit_op(e, OP_TYPE_ARITH_SITOFP, str_lit("arith.sitofp"),
            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
    return r;
}

static int64_t cmpi_pred_for(BinOp op, bool is_unsigned) {
    switch (op) {
        case OP_EQ: return 0; case OP_NE: return 1;
        case OP_LT: return is_unsigned ? 6 : 2;
        case OP_LE: return is_unsigned ? 7 : 3;
        case OP_GT: return is_unsigned ? 8 : 4;
        case OP_GE: return is_unsigned ? 9 : 5;
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
    bool is_f64;       // when is_float, the value has TY_F64 (else TY_F32)
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
    bool is_unsigned;  // integer val came from an `unsigned` C type; selects
                       // unsigned div/mod/shr (and comparisons) at use sites.
} EVal;

static EVal emit_expr(E *e, Scope *sc, Expr *ex);
static int64_t type_size(E *e, Type t);
static int64_t type_align(E *e, Type t);
static int64_t c_sizeof_expr(E *e, Scope *sc, Expr *ex);
// Compute the single-blob representation tinyc uses to lay a `union` out
// with all members overlapping at offset 0: an array of `*count` elements
// of `*elem` (an integer type whose width equals the union's alignment),
// sized to cover the union's largest member. Used by type_size,
// init_struct_types, emit_struct_zero and emit_struct_copy_path so that
// they all agree on the union layout.
static void union_blob_layout(E *e, StructDef *sd, MLIR_TypeHandle *elem,
                              int64_t *elem_size, int64_t *count);
static Type infer_expr_type(E *e, Scope *sc, Expr *ex);
static void emit_struct_zero(E *e, MLIR_ValueHandle base, MLIR_TypeHandle source_elem,
                             StructDef *sd, int32_t *prefix, size_t n_prefix);
static void emit_expr_for_side_effect(E *e, Scope *sc, Expr *ex);

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
    // Narrow `long long` / `uint64_t` initializers down to i32 when the
    // destination is a default-i32 local. Without this, a declaration
    // like `uint8_t byte = v & 0x7f;` (where `v` is i64) emits an
    // `llvm.store i64, ptr` into a 4-byte slot — the store overflows
    // into the adjacent stack slot, clobbering whatever lives next door
    // (commonly the i64 source variable). Use the existing coerce_eval
    // path to insert an `i32.wrap_i64` (a.k.a. trunc i64 → i32).
    if (v.is_i64) {
        return emit_trunci_i64_to_i32(e, v.val);
    }
    return v.val;
}

static MLIR_ValueHandle coerce_eval(E *e, EVal v, MLIR_TypeHandle want) {
    if (want == e->f64) {
        if (v.is_float) {
            if (v.is_f64) return v.val;
            return emit_fpext_f32_to_f64(e, v.val);
        }
        if (v.is_i64) {
            return emit_sitofp_f64(e, v.val);
        }
        return emit_sitofp_f64(e, v.val);
    }
    if (want == e->f32) {
        if (v.is_float) {
            if (v.is_f64) return emit_fptrunc_f64_to_f32(e, v.val);
            return v.val;
        }
        return emit_sitofp(e, v.val);
    }
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
        // Use unsigned widening when the source carried an `unsigned`
        // C type — otherwise an i32 value with its top bit set (e.g.
        // a wasm32 `size_t` of 2 GiB) would sign-extend to a huge
        // "negative" u64.
        return v.is_unsigned ? emit_extui_i32_to_i64(e, v.val)
                             : emit_extsi_i32_to_i64(e, v.val);
    }
    if (want == e->i32 && v.is_i64) {
        return emit_trunci_i64_to_i32(e, v.val);
    }
    if (want == e->i8) {
        // Storing into a `char*` (i8 element type): truncate any wider
        // integer value down to i8.
        if (v.is_i64) {
            MLIR_ValueHandle as_i32 = emit_trunci_i64_to_i32(e, v.val);
            return emit_trunci_to_i8(e, as_i32);
        }
        return emit_trunci_to_i8(e, v.val);
    }
    if (want == e->ptr) {
        if (v.is_ptr) return v.val;
        return emit_null_ptr(e);
    }
    return v.val;
}

static MLIR_ValueHandle emit_to_bool_i1(E *e, EVal v) {
    if (v.is_float) {
        MLIR_ValueHandle z = v.is_f64 ? emit_const_f64(e, 0.0) : emit_const_f32(e, 0.0);
        return emit_cmpf(e, /*one*/ 6, v.val, z);
    }
    if (v.is_ptr) {
        MLIR_ValueHandle z = emit_null_ptr(e);
        return emit_icmp_ptr(e, /*ne*/ 1, v.val, z);
    }
    if (v.is_i64) {
        MLIR_ValueHandle z = emit_const_i64(e, 0);
        return emit_cmpi(e, /*ne*/ 1, v.val, z);
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
    bool             is_unsigned;     // backing C type was `unsigned`
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
    if (lv.elem_ty == e->i8) {
        // Widen char/uint8 to int. `unsigned char` / `uint8_t` zero-extend
        // (lv.is_unsigned); signed/plain `char` sign-extend so negative
        // values stay negative for subsequent arithmetic.
        r.val = lv.is_unsigned ? emit_extui_i8_to_i32(e, r.val)
                               : emit_extsi_i8_to_i32(e, r.val);
    }
    r.is_float = (lv.elem_ty == e->f32 || lv.elem_ty == e->f64);
    r.is_f64 = (lv.elem_ty == e->f64);
    r.is_i64 = (lv.elem_ty == e->i64);
    r.is_ptr = (lv.elem_ty == e->ptr);
    r.is_unsigned = lv.is_unsigned;
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
        if (s && s->type.kind == TY_PTR_STRUCT && s->sdef) {
            // `p[i].f` where p is a struct pointer: load p, then GEP by index.
            MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
            r.base_ptr = emit_load_v(e, sym_addr(e, s), e->ptr);
            r.source_elem = find_struct_type(e, s->sdef);
            r.sd = s->sdef;
            sctx_push(e, &r, LLVM_GEP_DYN);
            r.dyn_index = idx_i32;
            r.ok = true;
            return r;
        }
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
    if (ex->kind == EX_INDEX) {
        // Generic `expr[i].f` where expr evaluates to a struct pointer
        // (e.g. `ht->buckets[i].f`). Evaluate the LHS as an EVal, then
        // GEP into the struct array.
        EVal v = emit_expr(e, sc, ex->lhs);
        if (v.is_ptr && v.sdef) {
            MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
            r.base_ptr = v.val;
            r.source_elem = find_struct_type(e, v.sdef);
            r.sd = v.sdef;
            sctx_push(e, &r, LLVM_GEP_DYN);
            r.dyn_index = idx_i32;
            r.ok = true;
            return r;
        }
        EMIT_ERR(e, "arr[i].f requires an array of struct or struct-pointer base");
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
        if (parent.sd->is_union) {
            // Crossing a union boundary into a nested struct member.
            // Emit a partial GEP to offset 0 of the union, then start
            // a fresh GEP rooted at the nested struct's type. This is
            // necessary because tinyc lays unions out as sequential
            // structs, so subsequent indices into the nested struct
            // would otherwise descend into the wrong union member's
            // LLVM type.
            sctx_push(e, &r, 0);
            MLIR_ValueHandle base = emit_gep(e, r.base_ptr, r.source_elem,
                                              r.const_path, r.n_const_path,
                                              NULL, 0);
            r.base_ptr = base;
            r.source_elem = find_struct_type(e, find_struct(e, ft.struct_name));
            r.const_path = NULL;
            r.n_const_path = 0;
            r.cap_const_path = 0;
            r.dyn_index = MLIR_INVALID_HANDLE;
            sctx_push(e, &r, 0);
        } else {
            sctx_push(e, &r, idx);
        }
        r.sd = find_struct(e, ft.struct_name);
        r.ok = (r.sd != NULL);
        return r;
    }
    // Fallback: a struct-returning call result, `f(...).field`. The
    // call yields an EVal with is_ptr=true and sdef set; we treat that
    // !llvm.ptr as the struct base.
    if (ex->kind == EX_CALL) {
        EVal v = emit_expr(e, sc, ex);
        if (v.is_ptr && v.sdef) {
            r.base_ptr = v.val;
            r.sd = v.sdef;
            r.source_elem = find_struct_type(e, v.sdef);
            sctx_push(e, &r, 0);
            r.ok = true;
            return r;
        }
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
            else if (s->type.kind == TY_F64) r.elem_ty = e->f64;
            else if (s->type.kind == TY_I64) r.elem_ty = e->i64;
            else if (s->type.kind == TY_PTR_STRUCT || s->type.kind == TY_PTR_I32 ||
                     s->type.kind == TY_PTR_CHAR || s->type.kind == TY_PTR_VOID ||
                     s->type.kind == TY_FNPTR || s->type.kind == TY_PTR_PTR)
                r.elem_ty = e->ptr;
            else r.elem_ty = e->i32;
            r.is_unsigned = s->type.int_unsigned;
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
                        MLIR_TypeHandle elem;
                        if (is_arr_f32) elem = e->f32;
                        else if (is_arr_pst || is_arr_pch) elem = e->ptr;
                        else if (ft.array_elem_is_i64) elem = e->i64;
                        else if (ft.array_elem_is_i8) elem = e->i8;
                        else elem = e->i32;
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
                        r.is_unsigned = ft.int_unsigned;
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
                                              : ft.ptr_is_f32 ? e->f32
                                              : ft.ptr_is_f64 ? e->f64
                                              : e->i32;
                        MLIR_ValueHandle iv = emit_expr_i32(e, sc, idx_a);
                        int32_t *gpath = arena_new_array(e->arena, int32_t, 1);
                        gpath[0] = LLVM_GEP_DYN;
                        MLIR_ValueHandle *gdyn = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                        gdyn[0] = iv;
                        r.base_ptr = emit_gep(e, base, elem, gpath, 1, gdyn, 1);
                        r.elem_ty = elem;
                        r.is_unsigned = ft.int_unsigned;
                        return r;
                    }
                }
            }
            // Array-of-char-pointers chained indexing:
            //   `const char *T[N]; ... T[i][j]`.
            // Step 1: GEP into the local array at index i to get the slot
            //         that holds the inner char*.
            // Step 2: load that slot to obtain the inner char*.
            // Step 3: GEP char* + j (stride 1) -> ptr-to-char.
            if (ex->lhs->kind == EX_INDEX &&
                ex->lhs->lhs->kind == EX_VAR) {
                Sym *s = lookup(e, sc, ex->lhs->lhs->name);
                if (s && (s->type.kind == TY_ARRAY_PTR_CHAR ||
                          s->type.kind == TY_ARRAY_PTR_STRUCT)) {
                    MLIR_ValueHandle i_v = emit_expr_i32(e, sc, ex->lhs->rhs);
                    MLIR_ValueHandle j_v = emit_expr_i32(e, sc, ex->rhs);
                    MLIR_TypeHandle arr_ty = MLIR_CreateTypeLLVMArray(
                        e->ctx, e->ptr, (uint64_t)s->type.array_len);
                    int32_t *path1 = arena_new_array(e->arena, int32_t, 2);
                    path1[0] = 0; path1[1] = LLVM_GEP_DYN;
                    MLIR_ValueHandle *dyn1 = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                    dyn1[0] = i_v;
                    MLIR_ValueHandle slot = emit_gep(e, sym_addr(e, s),
                        arr_ty, path1, 2, dyn1, 1);
                    MLIR_ValueHandle inner = emit_load_v(e, slot, e->ptr);
                    MLIR_TypeHandle elem = (s->type.kind == TY_ARRAY_PTR_CHAR)
                        ? e->i8 : e->ptr;
                    int32_t *path2 = arena_new_array(e->arena, int32_t, 1);
                    path2[0] = LLVM_GEP_DYN;
                    MLIR_ValueHandle *dyn2 = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                    dyn2[0] = j_v;
                    r.base_ptr = emit_gep(e, inner, elem, path2, 1, dyn2, 1);
                    r.elem_ty = elem;
                    return r;
                }
            }
            // Pointer-to-pointer chained indexing: pp[i][j] where pp is T**.
            // Step 1: GEP pp+i (stride sizeof(ptr)) -> ptr-to-ptr.
            // Step 2: load that to obtain the inner T*.
            // Step 3: GEP T*+j (stride sizeof(T)) -> ptr-to-T.
            if (ex->lhs->kind == EX_INDEX &&
                ex->lhs->lhs->kind == EX_VAR) {
                Sym *s = lookup(e, sc, ex->lhs->lhs->name);
                if (s && s->type.kind == TY_PTR_PTR) {
                    MLIR_ValueHandle i_v = emit_expr_i32(e, sc, ex->lhs->rhs);
                    MLIR_ValueHandle j_v = emit_expr_i32(e, sc, ex->rhs);
                    // pp[i]: GEP at base of pp (which holds T**), stride i.
                    MLIR_ValueHandle base_pp = emit_load_v(e, sym_addr(e, s), e->ptr);
                    int32_t *path1 = arena_new_array(e->arena, int32_t, 1);
                    path1[0] = LLVM_GEP_DYN;
                    MLIR_ValueHandle *dyn1 = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                    dyn1[0] = i_v;
                    MLIR_ValueHandle slot = emit_gep(e, base_pp, e->ptr, path1, 1, dyn1, 1);
                    // Load T* from that slot.
                    MLIR_ValueHandle inner = emit_load_v(e, slot, e->ptr);
                    // Decide element type of inner (T).
                    MLIR_TypeHandle elem = e->i8;
                    if (s->type.pointee) {
                        TypeKind pk = s->type.pointee->kind;
                        if (pk == TY_PTR_CHAR) elem = e->i8;
                        else if (pk == TY_PTR_I32) {
                            elem = s->type.pointee->ptr_is_i64 ? e->i64
                                 : s->type.pointee->ptr_is_f32 ? e->f32
                                 : s->type.pointee->ptr_is_f64 ? e->f64
                                 : e->i32;
                        }
                    }
                    int32_t *path2 = arena_new_array(e->arena, int32_t, 1);
                    path2[0] = LLVM_GEP_DYN;
                    MLIR_ValueHandle *dyn2 = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                    dyn2[0] = j_v;
                    r.base_ptr = emit_gep(e, inner, elem, path2, 1, dyn2, 1);
                    r.elem_ty = elem;
                    return r;
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
                // <struct>.field[s][i] / <ptr>->field[s][i] where field is T**.
                if (ex->lhs->kind == EX_INDEX &&
                    ex->lhs->lhs->kind == EX_FIELD) {
                    Expr *fld = ex->lhs->lhs;
                    Expr *vex = fld->lhs;
                    bool is_arrow = (vex->kind == EX_DEREF && vex->lhs->kind == EX_VAR);
                    bool is_dot = (vex->kind == EX_VAR);
                    if (is_arrow || is_dot) {
                        Expr *vv = is_arrow ? vex->lhs : vex;
                        Sym *vs = lookup(e, sc, vv->name);
                        StructDef *sd = NULL;
                        MLIR_ValueHandle parent_base = MLIR_INVALID_HANDLE;
                        if (vs) {
                            if (is_arrow && vs->type.kind == TY_PTR_STRUCT) {
                                sd = find_struct(e, vs->type.struct_name);
                                parent_base = emit_load_v(e, sym_addr(e, vs), e->ptr);
                            } else if (is_dot && vs->type.kind == TY_STRUCT) {
                                sd = find_struct(e, vs->type.struct_name);
                                parent_base = sym_addr(e, vs);
                            }
                        }
                        if (sd) {
                            size_t fidx = 0; bool found = false;
                            Type fldt = (Type){0};
                            for (size_t k = 0; k < sd->fields.size; k++) {
                                if (str_eq(sd->fields.data[k].name, fld->name)) {
                                    fidx = k; fldt = sd->fields.data[k].type; found = true; break;
                                }
                            }
                            if (found && fldt.kind == TY_PTR_PTR) {
                                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                                int32_t *fpath = arena_new_array(e->arena, int32_t, 2);
                                fpath[0] = 0; fpath[1] = (int32_t)fidx;
                                MLIR_ValueHandle field_addr = emit_gep(e, parent_base,
                                    st_ty, fpath, 2, NULL, 0);
                                MLIR_ValueHandle pp = emit_load_v(e, field_addr, e->ptr);
                                MLIR_ValueHandle sv = emit_expr_i32(e, sc, ex->lhs->rhs);
                                int32_t *p1 = arena_new_array(e->arena, int32_t, 1);
                                p1[0] = LLVM_GEP_DYN;
                                MLIR_ValueHandle *d1 = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                                d1[0] = sv;
                                MLIR_ValueHandle slot = emit_gep(e, pp, e->ptr, p1, 1, d1, 1);
                                MLIR_ValueHandle inner = emit_load_v(e, slot, e->ptr);
                                MLIR_TypeHandle elem = e->i8;
                                if (fldt.pointee) {
                                    TypeKind pk = fldt.pointee->kind;
                                    if (pk == TY_PTR_CHAR) elem = e->i8;
                                    else if (pk == TY_PTR_I32) {
                                        elem = fldt.pointee->ptr_is_i64 ? e->i64
                                             : fldt.pointee->ptr_is_f32 ? e->f32
                                             : fldt.pointee->ptr_is_f64 ? e->f64
                                             : e->i32;
                                    } else if (pk == TY_PTR_VOID || pk == TY_PTR_STRUCT ||
                                               pk == TY_FNPTR || pk == TY_PTR_PTR) {
                                        elem = e->ptr;
                                    }
                                }
                                MLIR_ValueHandle iv = emit_expr_i32(e, sc, ex->rhs);
                                int32_t *p2 = arena_new_array(e->arena, int32_t, 1);
                                p2[0] = LLVM_GEP_DYN;
                                MLIR_ValueHandle *d2 = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                                d2[0] = iv;
                                r.base_ptr = emit_gep(e, inner, elem, p2, 1, d2, 1);
                                r.elem_ty = elem;
                                return r;
                            }
                        }
                    }
                }
                // (*p)[i] where p is T**: load p (get T**), load again (get
                // T*), then GEP by i with stride sizeof(T).
                if (ex->lhs->kind == EX_DEREF &&
                    ex->lhs->lhs->kind == EX_VAR) {
                    Sym *s = lookup(e, sc, ex->lhs->lhs->name);
                    if (s && s->type.kind == TY_PTR_PTR) {
                        MLIR_ValueHandle iv = emit_expr_i32(e, sc, ex->rhs);
                        MLIR_ValueHandle outer = emit_load_v(e, sym_addr(e, s), e->ptr);
                        MLIR_ValueHandle inner = emit_load_v(e, outer, e->ptr);
                        MLIR_TypeHandle elem = e->i8;
                        if (s->type.pointee) {
                            TypeKind pk = s->type.pointee->kind;
                            if (pk == TY_PTR_CHAR) elem = e->i8;
                            else if (pk == TY_PTR_I32) {
                                elem = s->type.pointee->ptr_is_i64 ? e->i64
                                     : s->type.pointee->ptr_is_f32 ? e->f32
                                     : s->type.pointee->ptr_is_f64 ? e->f64
                                     : e->i32;
                            } else if (pk == TY_PTR_VOID || pk == TY_PTR_STRUCT ||
                                       pk == TY_FNPTR || pk == TY_PTR_PTR) {
                                elem = e->ptr;
                            }
                        }
                        int32_t *gpath = arena_new_array(e->arena, int32_t, 1);
                        gpath[0] = LLVM_GEP_DYN;
                        MLIR_ValueHandle *gdyn = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                        gdyn[0] = iv;
                        r.base_ptr = emit_gep(e, inner, elem, gpath, 1, gdyn, 1);
                        r.elem_ty = elem;
                        return r;
                    }
                }
                EMIT_ERR(e, "only simple-array indexing is supported");
                return r;
            }
            Sym *s = lookup(e, sc, ex->lhs->name);
            if (!s) {
                EMIT_ERR(e, "undefined array: {}", ex->lhs->name);
                return r;
            }
            if (s->type.kind == TY_ARRAY_STRUCT) {
                // For lvalue context that takes the *address* of arr[i]
                // (e.g. `&arr[i]`), GEP into the local array. Field access
                // (`arr[i].f`) goes through other paths that build the GEP
                // themselves; we still keep the original guard for that.
                StructDef *sd = find_struct(e, s->type.struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct in array");
                    return r;
                }
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
                r.base_ptr = sym_addr(e, s);
                r.source_elem = MLIR_CreateTypeLLVMArray(e->ctx, st_ty,
                    (uint64_t)s->type.array_len);
                int32_t *path = arena_new_array(e->arena, int32_t, 2);
                path[0] = 0; path[1] = LLVM_GEP_DYN;
                r.const_path = path; r.n_const_path = 2;
                r.dyn_index = idx_i32;
                r.elem_ty = st_ty;
                return r;
            }
            if (s->type.kind == TY_PTR_STRUCT) {
                // p[i] for a struct pointer parameter / local: load the
                // pointer then GEP by the struct stride.
                StructDef *sd = find_struct(e, s->type.struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct ptr");
                    return r;
                }
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
                MLIR_ValueHandle base = emit_load_v(e, sym_addr(e, s), e->ptr);
                int32_t *path = arena_new_array(e->arena, int32_t, 1);
                path[0] = LLVM_GEP_DYN;
                MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                dyn[0] = idx_i32;
                r.base_ptr = emit_gep(e, base, st_ty, path, 1, dyn, 1);
                r.elem_ty = st_ty;
                return r;
            }
            // Pointer indexing: p[i] for int* / char* — GEP %p[%i].
            if (s->type.kind == TY_PTR_I32 || s->type.kind == TY_PTR_CHAR) {
                MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
                MLIR_ValueHandle base = emit_load_v(e, sym_addr(e, s), e->ptr);
                MLIR_TypeHandle elem = (s->type.kind == TY_PTR_CHAR) ? e->i8
                                      : s->type.ptr_is_i64 ? e->i64
                                      : s->type.ptr_is_f32 ? e->f32
                                      : s->type.ptr_is_f64 ? e->f64
                                      : e->i32;
                int32_t *path = arena_new_array(e->arena, int32_t, 1);
                path[0] = LLVM_GEP_DYN;
                MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                dyn[0] = idx_i32;
                r.base_ptr = emit_gep(e, base, elem, path, 1, dyn, 1);
                r.elem_ty = elem;
                r.is_unsigned = s->type.int_unsigned;
                return r;
            }
            // Pointer-to-pointer indexing: pp[i] for T** (e.g. char **argv).
            // Each element is a !llvm.ptr; GEP with stride 8 and load yields
            // a single-level T*.
            if (s->type.kind == TY_PTR_PTR) {
                MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
                MLIR_ValueHandle base = emit_load_v(e, sym_addr(e, s), e->ptr);
                int32_t *path = arena_new_array(e->arena, int32_t, 1);
                path[0] = LLVM_GEP_DYN;
                MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                dyn[0] = idx_i32;
                r.base_ptr = emit_gep(e, base, e->ptr, path, 1, dyn, 1);
                r.elem_ty = e->ptr;
                return r;
            }
            if (s->type.kind == TY_ARRAY_PTR_CHAR ||
                s->type.kind == TY_ARRAY_PTR_STRUCT) {
                MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
                r.base_ptr = sym_addr(e, s);
                r.source_elem = MLIR_CreateTypeLLVMArray(e->ctx, e->ptr, (uint64_t)s->type.array_len);
                int32_t *path = arena_new_array(e->arena, int32_t, 2);
                path[0] = 0; path[1] = LLVM_GEP_DYN;
                r.const_path = path; r.n_const_path = 2;
                r.dyn_index = idx_i32;
                r.elem_ty = e->ptr;
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
            MLIR_TypeHandle aelem = s->type.array_elem_is_i64 ? e->i64
                                  : s->type.array_elem_is_i8  ? e->i8
                                  : e->i32;
            r.base_ptr = sym_addr(e, s);
            r.source_elem = MLIR_CreateTypeLLVMArray(e->ctx, aelem, (uint64_t)s->type.array_len);
            int32_t *path = arena_new_array(e->arena, int32_t, 2);
            path[0] = 0; path[1] = LLVM_GEP_DYN;
            r.const_path = path; r.n_const_path = 2;
            r.dyn_index = idx_i32;
            r.elem_ty = aelem;
            r.is_unsigned = s->type.int_unsigned;
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
                else if (s->type.ptr_is_i64) r.elem_ty = e->i64;
                else if (s->type.ptr_is_f32) r.elem_ty = e->f32;
                else if (s->type.ptr_is_f64) r.elem_ty = e->f64;
                else r.elem_ty = e->i32;
                r.is_unsigned = s->type.int_unsigned;
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
            r.is_unsigned = v.is_unsigned;
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
                ft.kind != TY_F64 &&
                ft.kind != TY_PTR_STRUCT && ft.kind != TY_PTR_I32 &&
                ft.kind != TY_PTR_CHAR && ft.kind != TY_PTR_VOID &&
                ft.kind != TY_FNPTR && ft.kind != TY_PTR_PTR) {
                EMIT_ERR(e, "field {} is not a scalar lvalue", ex->name);
                return r;
            }
            sctx_push(e, &parent, parent.sd->is_union ? 0 : idx);
            r.base_ptr = parent.base_ptr;
            r.source_elem = parent.source_elem;
            r.const_path = parent.const_path;
            r.n_const_path = parent.n_const_path;
            r.dyn_index = parent.dyn_index;
            if (ft.kind == TY_F32) r.elem_ty = e->f32;
            else if (ft.kind == TY_F64) r.elem_ty = e->f64;
            else if (ft.kind == TY_I64) r.elem_ty = e->i64;
            else if (ft.kind == TY_PTR_STRUCT || ft.kind == TY_PTR_I32 ||
                     ft.kind == TY_PTR_CHAR || ft.kind == TY_PTR_VOID ||
                     ft.kind == TY_FNPTR || ft.kind == TY_PTR_PTR) r.elem_ty = e->ptr;
            else r.elem_ty = e->i32;
            r.is_unsigned = ft.int_unsigned;
            return r;
        }
        default:
            EMIT_ERR(e, "invalid lvalue");
            return r;
    }
}

static void unify_numeric(E *e, EVal *a, EVal *b) {
    if (a->is_float || b->is_float) {
        // Promote integer side to a matching float type first.
        if (!a->is_float) {
            if (a->is_i64) { a->val = emit_trunci_i64_to_i32(e, a->val); a->is_i64 = false; }
            if (b->is_f64) { a->val = emit_sitofp_f64(e, a->val); a->is_f64 = true; }
            else           { a->val = emit_sitofp(e, a->val); }
            a->is_float = true;
        }
        if (!b->is_float) {
            if (b->is_i64) { b->val = emit_trunci_i64_to_i32(e, b->val); b->is_i64 = false; }
            if (a->is_f64) { b->val = emit_sitofp_f64(e, b->val); b->is_f64 = true; }
            else           { b->val = emit_sitofp(e, b->val); }
            b->is_float = true;
        }
        // If one is f32 and the other f64, fpext the f32 side.
        if (a->is_f64 != b->is_f64) {
            if (!a->is_f64) { a->val = emit_fpext_f32_to_f64(e, a->val); a->is_f64 = true; }
            if (!b->is_f64) { b->val = emit_fpext_f32_to_f64(e, b->val); b->is_f64 = true; }
        }
        return;
    }
    if (a->is_i64 == b->is_i64) {
        bool u = a->is_unsigned || b->is_unsigned;
        a->is_unsigned = u; b->is_unsigned = u;
        return;
    }
    if (!a->is_i64) {
        a->val = a->is_unsigned ? emit_extui_i32_to_i64(e, a->val)
                                : emit_extsi_i32_to_i64(e, a->val);
        a->is_i64 = true;
    }
    if (!b->is_i64) {
        b->val = b->is_unsigned ? emit_extui_i32_to_i64(e, b->val)
                                : emit_extsi_i32_to_i64(e, b->val);
        b->is_i64 = true;
    }
    bool u = a->is_unsigned || b->is_unsigned;
    a->is_unsigned = u; b->is_unsigned = u;
}

// ----- struct-by-pointer helpers -----

// Resolve a struct-typed call argument into the source !llvm.ptr.
// Accepts:
//   - EX_VAR struct          -> sym->addr
//   - EX_VAR struct*         -> load(sym->addr)
//   - EX_ADDR(EX_VAR struct) -> sym->addr
//   - any expression yielding a pointer (struct* field load, malloc cast,
//     null, etc.) -> the EVal's !llvm.ptr value
static void emit_struct_copy(E *e, MLIR_ValueHandle dst, MLIR_ValueHandle src, StructDef *sd);
static MLIR_ValueHandle resolve_struct_source(E *e, Scope *sc, Expr *arg, StructDef **out_sd) {
    Expr *target = arg;
    if (arg->kind == EX_ADDR) {
        // `&<var>` — handled below.
        // `&arr[i]` where arr is array-of-struct: GEP to element address.
        if (arg->lhs->kind == EX_INDEX && arg->lhs->lhs->kind == EX_VAR) {
            Sym *s = lookup(e, sc, arg->lhs->lhs->name);
            if (s && s->type.kind == TY_ARRAY_STRUCT && s->sdef) {
                MLIR_ValueHandle idx = emit_expr_i32(e, sc, arg->lhs->rhs);
                MLIR_TypeHandle st_ty = find_struct_type(e, s->sdef);
                MLIR_TypeHandle arr_ty = MLIR_CreateTypeLLVMArray(
                    e->ctx, st_ty, (uint64_t)s->type.array_len);
                int32_t path[2] = {0, LLVM_GEP_DYN};
                MLIR_ValueHandle dyn[1] = {idx};
                MLIR_ValueHandle p = emit_gep(e, sym_addr(e, s), arr_ty, path, 2, dyn, 1);
                *out_sd = s->sdef;
                return p;
            }
            // `&p[i]` where p is struct*: load p, then GEP by i (stride sizeof(struct)).
            if (s && s->type.kind == TY_PTR_STRUCT && s->sdef) {
                MLIR_ValueHandle idx = emit_expr_i32(e, sc, arg->lhs->rhs);
                MLIR_ValueHandle base = emit_load_v(e, sym_addr(e, s), e->ptr);
                MLIR_TypeHandle st_ty = find_struct_type(e, s->sdef);
                int32_t path[1] = {LLVM_GEP_DYN};
                MLIR_ValueHandle dyn[1] = {idx};
                MLIR_ValueHandle p = emit_gep(e, base, st_ty, path, 1, dyn, 1);
                *out_sd = s->sdef;
                return p;
            }
        }
        if (arg->lhs->kind != EX_VAR) {
            // Generic case: `&<lvalue-chain>` where the lvalue resolves
            // to a struct-typed slot reachable via FIELD/ARROW/INDEX
            // descent (e.g. `&st->scopes[i]`).
            Type t = infer_expr_type(e, sc, arg->lhs);
            if (t.kind == TY_STRUCT) {
                StructDef *sd = find_struct(e, t.struct_name);
                if (sd) {
                    SCtx c = walk_struct_lhs(e, sc, arg->lhs);
                    if (c.ok && c.sd == sd) {
                        LVal lv = (LVal){0};
                        lv.base_ptr = c.base_ptr;
                        lv.source_elem = c.source_elem;
                        lv.const_path = c.const_path;
                        lv.n_const_path = c.n_const_path;
                        lv.dyn_index = c.dyn_index;
                        *out_sd = sd;
                        return lval_address(e, lv);
                    }
                }
            }
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
    // `*<expr>` where <expr> is a struct pointer: yield the loaded
    // !llvm.ptr value; the caller treats it as the struct source for
    // a by-value copy.
    if (arg->kind == EX_DEREF) {
        Type inner = infer_expr_type(e, sc, arg->lhs);
        if (inner.kind == TY_PTR_STRUCT) {
            StructDef *sd = find_struct(e, inner.struct_name);
            if (sd) {
                EVal pv = emit_expr(e, sc, arg->lhs);
                if (pv.is_ptr) {
                    *out_sd = sd;
                    return pv.val;
                }
            }
        }
    }
    // Struct lvalue chain (e.g. `ht->buckets[i].value` where `.value`
    // is itself a struct). walk_struct_lhs descends FIELD/INDEX/ARROW
    // chains and stops at struct-typed fields, yielding a base_ptr +
    // path that lval_address can resolve to the struct's address.
    if (arg->kind == EX_FIELD || arg->kind == EX_INDEX) {
        Type t = infer_expr_type(e, sc, arg);
        if (t.kind == TY_STRUCT) {
            StructDef *sd = find_struct(e, t.struct_name);
            if (sd) {
                SCtx c = walk_struct_lhs(e, sc, arg);
                if (c.ok && c.sd == sd) {
                    LVal lv = (LVal){0};
                    lv.base_ptr = c.base_ptr;
                    lv.source_elem = c.source_elem;
                    lv.const_path = c.const_path;
                    lv.n_const_path = c.n_const_path;
                    lv.dyn_index = c.dyn_index;
                    *out_sd = sd;
                    return lval_address(e, lv);
                }
            }
        }
    }
    // Ternary returning a struct: `cond ? a : b` where both arms are
    // struct-typed of the same struct. Allocate one slot, copy in each
    // branch, return the slot.
    if (arg->kind == EX_TERNARY) {
        Type t = infer_expr_type(e, sc, arg->rhs);
        if (t.kind == TY_STRUCT) {
            StructDef *sd = find_struct(e, t.struct_name);
            if (sd) {
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                MLIR_ValueHandle save_ptr = MLIR_INVALID_HANDLE;
                MLIR_BlockHandle save_blk = e->cur_block;
                MLIR_ValueHandle slot = emit_alloca(e, st_ty);
                EVal cv = emit_expr(e, sc, arg->lhs);
                MLIR_ValueHandle cb = emit_to_bool_i1(e, cv);
                MLIR_BlockHandle then_blk = new_cfg_block(e);
                MLIR_BlockHandle else_blk = new_cfg_block(e);
                MLIR_BlockHandle merge_blk = new_cfg_block(e);
                e->cur_block = then_blk; e->terminated = false;
                StructDef *tsd = NULL;
                MLIR_ValueHandle tsrc = resolve_struct_source(e, sc, arg->rhs, &tsd);
                if (tsrc != MLIR_INVALID_HANDLE && tsd == sd) {
                    emit_struct_copy(e, slot, tsrc, sd);
                }
                emit_branch(e, merge_blk);
                e->cur_block = else_blk; e->terminated = false;
                StructDef *esd = NULL;
                MLIR_ValueHandle esrc = resolve_struct_source(e, sc, arg->lvalue, &esd);
                if (esrc != MLIR_INVALID_HANDLE && esd == sd) {
                    emit_struct_copy(e, slot, esrc, sd);
                }
                emit_branch(e, merge_blk);
                e->cur_block = save_blk; e->terminated = false;
                emit_cond_branch(e, cb, then_blk, else_blk);
                e->cur_block = merge_blk; e->terminated = false;
                (void)save_ptr;
                *out_sd = sd;
                return slot;
            }
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
    if (sd->is_union) {
        // The union's llvm body is a single blob field (index 0) of
        // `count` alignment-wide integers. Copy the blob element by element.
        MLIR_TypeHandle uelem; int64_t ues, uc;
        union_blob_layout(e, sd, &uelem, &ues, &uc);
        for (int64_t j = 0; j < uc; j++) {
            size_t total = n_prefix + 2;
            int32_t *p2 = arena_new_array(e->arena, int32_t, total);
            for (size_t k = 0; k < n_prefix; k++) p2[k] = prefix[k];
            p2[n_prefix] = 0;
            p2[n_prefix + 1] = (int32_t)j;
            MLIR_ValueHandle sp = emit_gep(e, src, source_elem, p2, total, NULL, 0);
            MLIR_ValueHandle val = emit_load_v(e, sp, uelem);
            MLIR_ValueHandle dp = emit_gep(e, dst, source_elem, p2, total, NULL, 0);
            emit_store_v(e, val, dp);
        }
        return;
    }
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
            else if (ft.array_elem_is_i64) elem = e->i64;
            else if (ft.array_elem_is_i8) elem = e->i8;
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
    // Variadic struct args may expand into multiple i64 words; account for that.
    if (sig->is_variadic) {
        for (size_t i = sig->n_params; i < args.size; i++) {
            Type at = infer_expr_type(e, sc, args.data[i]);
            if (at.kind == TY_STRUCT) {
                int64_t bytes = type_size(e, at);
                int64_t words = (bytes + 7) / 8;
                if (words > 1) n_in += (words - 1);
            }
        }
    }
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
    // Variadic-portion arguments: apply C default argument promotion.
    // float -> double, integers narrower than int are already int (we
    // model all narrow ints as i32 in tinyC). Struct-by-value args are
    // unpacked into i64 words to match the helper-side va_arg layout.
    for (size_t i = sig->n_params; i < args.size; i++) {
        Type at = infer_expr_type(e, sc, args.data[i]);
        if (at.kind == TY_STRUCT) {
            StructDef *asd = NULL;
            MLIR_ValueHandle src = resolve_struct_source(e, sc, args.data[i], &asd);
            if (src == MLIR_INVALID_HANDLE || !asd) {
                EMIT_ERR(e, "variadic struct argument: cannot resolve");
                continue;
            }
            int64_t bytes = type_size(e, at);
            int64_t words = (bytes + 7) / 8;
            MLIR_TypeHandle warr = MLIR_CreateTypeLLVMArray(e->ctx, e->i64, (uint64_t)words);
            for (int64_t w = 0; w < words; w++) {
                int32_t path[2] = {0, (int32_t)w};
                MLIR_ValueHandle wp = emit_gep(e, src, warr, path, 2, NULL, 0);
                MLIR_ValueHandle wv = emit_load_v(e, wp, e->i64);
                ops[op_off++] = wv;
            }
            continue;
        }
        EVal v = emit_expr(e, sc, args.data[i]);
        if (v.is_float && !v.is_f64) {
            v.val = emit_fpext_f32_to_f64(e, v.val);
            v.is_f64 = true;
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
static bool ast_fold_int(E *e, Scope *sc, Expr *ex, int64_t *out) {
    if (!ex) return false;
    switch (ex->kind) {
        case EX_INT:
            *out = (int64_t)(int32_t)ex->int_value;
            return true;
        case EX_SIZEOF: {
            Type ty = (Type){0};
            if (ex->sizeof_is_expr) {
                if (!sc) return false;
                if (ex->lhs && ex->lhs->kind == EX_STR) {
                    // sizeof("string literal") yields the array size
                    // including the trailing NUL, matching standard C.
                    *out = (int64_t)ex->lhs->name.size;
                    return true;
                }
                ty = infer_expr_type(e, sc, ex->lhs);
                if (ty.kind == TY_VOID) return false;
                *out = (int64_t)(int32_t)c_sizeof_expr(e, sc, ex->lhs);
                return true;
            } else {
                if (ex->cast_type.kind == TY_VOID) return false;
                ty = ex->cast_type;
            }
            *out = (int64_t)(int32_t)type_size(e, ty);
            return true;
        }
        case EX_UN: {
            int64_t a;
            if (!ast_fold_int(e, sc, ex->lhs, &a)) return false;
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
            if (!ast_fold_int(e, sc, ex->lhs, &a)) return false;
            if (!ast_fold_int(e, sc, ex->rhs, &b)) return false;
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
            if (!ast_fold_int(e, sc, ex->lhs, &c)) return false;
            Expr *pick = ((int32_t)c != 0) ? ex->rhs : ex->lvalue;
            return ast_fold_int(e, sc, pick, out);
        }
        default:
            return false;
    }
}

// Resolve a deferred array-length expression on `t` (set when the
// parser saw `[<const-expr>]`) into the integer `array_len` slot, in
// the given scope. No-op when array_len_expr is NULL or array_len is
// already set. The expression must constant-fold to a positive int32;
// otherwise we leave array_len at 0 and let the caller error.
static void resolve_array_len(E *e, Scope *sc, Type *t) {
    if (!t->array_len_expr) return;
    if (t->array_len > 0) return;
    int64_t v = 0;
    if (!ast_fold_int(e, sc, t->array_len_expr, &v)) {
        EMIT_ERR(e, "array length is not a constant integer expression");
        return;
    }
    if (v <= 0) {
        EMIT_ERR(e, "array length must be positive (got {})", v);
        return;
    }
    t->array_len = v;
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
    // C array decay: in _Generic, an array operand decays to a pointer.
    if (ct.kind == TY_ARRAY_I32 || ct.kind == TY_ARRAY_F32) {
        Type d = (Type){0};
        d.kind = TY_PTR_CHAR; // char[] is the typical case (string, buffer)
        // For int[] we'd want TY_PTR_I32; but in practice corec uses char[].
        // Still, prefer TY_PTR_I32 if int_bits suggests int.
        if (ct.int_bits != 0 && ct.int_bits != 8) d.kind = TY_PTR_I32;
        ct = d;
    } else if (ct.kind == TY_ARRAY_STRUCT || ct.kind == TY_ARRAY_PTR_STRUCT ||
               ct.kind == TY_ARRAY_PTR_CHAR) {
        Type d = (Type){0};
        d.kind = TY_PTR_VOID;
        ct = d;
    }
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
    // `__func__` is a magic predefined identifier carrying the enclosing
    // function's name as a NUL-terminated string literal. Synthesize the
    // string from the current FuncSig at emit time.
    if (ex->kind == EX_VAR &&
        ex->name.size == 8 &&
        memcmp(ex->name.str, "__func__", 8) == 0 &&
        !lookup(e, sc, ex->name)) {
        string fname = e->cur_sig ? e->cur_sig->name : str_lit("");
        // Build a NUL-terminated copy in arena and intern.
        char *buf = arena_alloc(e->arena, fname.size + 1);
        for (size_t i = 0; i < fname.size; i++) buf[i] = fname.str[i];
        buf[fname.size] = '\0';
        string bytes = (string){.str = buf, .size = fname.size + 1};
        string sym = intern_string(e, bytes);
        r.val = emit_addressof(e, sym);
        r.is_ptr = true;
        r.is_str = true;
        return r;
    }
    // Pre-pass: literal-only integer subtree -> a single arith.constant.
    // Only kicks in for nodes that ast_fold_int actually folds; anything
    // with a side effect or a non-int operand falls through unchanged.
    if (ex->kind == EX_BIN || ex->kind == EX_UN || ex->kind == EX_TERNARY) {
        int64_t fv;
        if (ast_fold_int(e, sc, ex, &fv)) {
            r.val = emit_const_i32(e, fv);
            return r;
        }
    }
    switch (ex->kind) {
        case EX_GENERIC:
            // Handled above; listed here only to silence -Wswitch.
            break;
        case EX_INT:
            if (ex->is_i64) {
                r.val = emit_const_i64(e, ex->int_value);
                r.is_i64 = true;
            } else {
                r.val = emit_const_i32(e, ex->int_value);
            }
            return r;
        case EX_FLOAT:
            if (ex->is_f64) {
                r.val = emit_const_f64(e, ex->float_value);
                r.is_float = true; r.is_f64 = true;
            } else {
                r.val = emit_const_f32(e, ex->float_value);
                r.is_float = true;
            }
            return r;
        case EX_NULL:
            r.val = emit_null_ptr(e); r.is_ptr = true; r.sdef = NULL; return r;
        case EX_VA_ARG: {
            // MLIR's LLVM dialect does not expose a `va_arg` op (only the
            // start/end/copy intrinsics) — and clang traditionally lowers
            // va_arg manually because its IR variant is buggy on x86_64.
            // We side-step both problems by calling small runtime helpers
            // that wrap stdarg.h; the helpers
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
            if (tk == TY_STRUCT) {
                // va_arg of a struct: alloca, then call a generic helper
                // that copies sizeof(struct) bytes (rounded up to 8) out
                // of the va_list into our buffer.
                StructDef *sd = find_struct(e, ex->cast_type.struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct in va_arg");
                    r.val = emit_const_i32(e, 0); return r;
                }
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                MLIR_ValueHandle out = emit_alloca(e, st_ty);
                MLIR_ValueHandle ap_ptr = sym_addr(e, sy);
                MLIR_ValueHandle sz = emit_const_i64(e, (int64_t)type_size(e, ex->cast_type));
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 3);
                ops[0] = ap_ptr; ops[1] = out; ops[2] = sz;
                MLIR_AttributeHandle ca = MLIR_CreateAttributeSymbolRef(
                    e->ctx, str_lit("callee"), str_lit("tinyc_va_arg_struct"));
                MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1);
                as[0] = ca;
                emit_op(e, OP_TYPE_FUNC_CALL, str_lit("func.call"),
                        NULL, 0, NULL, 0, ops, 3, as, 1, NULL, 0);
                e->need_va_arg_helpers = true;
                e->need_va_arg_struct = true;
                r.val = out;
                r.is_ptr = true;
                r.sdef = sd;
                return r;
            }
            string helper;
            MLIR_TypeHandle rt;
            switch (tk) {
                case TY_I32:        helper = str_lit("tinyc_va_arg_i32"); rt = e->i32; break;
                case TY_I64:        helper = str_lit("tinyc_va_arg_i64"); rt = e->i64; break;
                case TY_F64:        helper = str_lit("tinyc_va_arg_f64"); rt = e->f64; break;
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
            r.is_float = (tk == TY_F64);
            r.is_f64 = (tk == TY_F64);
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
            bool any_f64   = av.is_f64   || bv.is_f64;
            bool any_ptr   = av.is_ptr   || bv.is_ptr;
            bool any_i64   = av.is_i64   || bv.is_i64;
            MLIR_TypeHandle rty = any_ptr ? e->ptr : (any_float ? (any_f64 ? e->f64 : e->f32) : (any_i64 ? e->i64 : e->i32));

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
            r.is_f64   = any_f64;
            r.is_ptr   = any_ptr;
            r.is_i64   = any_i64;
            r.is_str   = av.is_str && bv.is_str;
            r.sdef     = av.sdef ? av.sdef : bv.sdef;
            return r;
        }
        case EX_SIZEOF: {
            Type ty = (Type){0};
            if (ex->sizeof_is_expr) {
                if (ex->lhs && ex->lhs->kind == EX_STR) {
                    r.val = emit_const_i32(e, (int32_t)ex->lhs->name.size);
                    return r;
                }
                ty = infer_expr_type(e, sc, ex->lhs);
                if (ty.kind == TY_VOID) {
                    EMIT_ERR(e, "sizeof of unsupported expression");
                    r.val = emit_const_i32(e, 1); return r;
                }
                r.val = emit_const_i32(e, c_sizeof_expr(e, sc, ex->lhs)); return r;
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
                // float-target casts (from int or float).
                if (ck == TY_F64) {
                    v.val = coerce_eval(e, v, e->f64);
                    v.is_float = true; v.is_f64 = true; v.is_i64 = false;
                    return v;
                }
                if (ck == TY_F32) {
                    v.val = coerce_eval(e, v, e->f32);
                    v.is_float = true; v.is_f64 = false; v.is_i64 = false;
                    return v;
                }
                // int-target casts.
                if (v.is_float) {
                    // float -> int via fptosi (i32) and optional ext to i64.
                    if (ck == TY_I64) {
                        v.val = coerce_eval(e, v, e->i64);
                        v.is_i64 = true;
                    } else {
                        v.val = coerce_eval(e, v, e->i32);
                        v.is_i64 = false;
                    }
                    v.is_float = false; v.is_f64 = false;
                    return v;
                }
                bool want_i64 = (ck == TY_I64);
                if (want_i64 && !v.is_i64) {
                    // Use unsigned widening when the source carried an
                    // `unsigned` C type — otherwise an i32 value with
                    // its top bit set (e.g. `(int64_t)<uint32_t>` with
                    // bit 31 set) sign-extends to a huge "negative"
                    // i64. The widening decision uses the SOURCE
                    // signedness; the result's signedness is the cast
                    // target's (set below).
                    v.val = v.is_unsigned ? emit_extui_i32_to_i64(e, v.val)
                                          : emit_extsi_i32_to_i64(e, v.val);
                    v.is_i64 = true;
                } else if (!want_i64 && v.is_i64) {
                    v.val = emit_trunci_i64_to_i32(e, v.val);
                    v.is_i64 = false;
                }
                // A cast to an unsigned integer type yields an unsigned
                // value: subsequent comparisons / div / shr must use the
                // unsigned form (e.g. `(unsigned long)x >= CONST`).
                v.is_unsigned = ex->cast_type.int_unsigned;
                return v;
            }
            // Pointer-to-integer cast: emit llvm.ptrtoint, then optionally
            // truncate to i32. Used for `(uintptr_t)p` / `(long)p` / etc.
            if (!ck_is_ptr && v.is_ptr) {
                MLIR_ValueHandle iv = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                               e->i64, ssa_name(e), eloc(e, 0));
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i64;
                MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = iv;
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v.val;
                emit_op(e, OP_TYPE_LLVM_PTRTOINT, str_lit("llvm.ptrtoint"),
                        rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
                v.val = iv;
                v.is_ptr = false;
                v.is_void_ptr = false;
                v.sdef = NULL;
                v.is_str = false;
                if (ck == TY_I64) {
                    v.is_i64 = true;
                } else {
                    v.val = emit_trunci_i64_to_i32(e, v.val);
                    v.is_i64 = false;
                }
                // `(uintptr_t)p` / `(unsigned long)p` produce an unsigned
                // integer: select unsigned comparisons / div / shr below.
                v.is_unsigned = ex->cast_type.int_unsigned;
                return v;
            }
            // Pointer-to-pointer cast: opaque !llvm.ptr is universal, so
            // just evaluate the operand. We tag the result type from
            // cast_type for downstream consumers.
            if (!v.is_ptr) {
                // Integer-to-pointer cast: literal 0 -> null pointer
                // (used for `(T*)0` null sentinels in C). Any other
                // integer operand is materialized via llvm.inttoptr
                // (used in corec for `(struct buddy_block *)addr` where
                // `addr` is a uintptr_t).
                if (ex->lhs->kind == EX_INT && ex->lhs->int_value == 0) {
                    v.val = emit_null_ptr(e);
                } else if (!v.is_float) {
                    // Ensure the operand is i64 for inttoptr.
                    MLIR_ValueHandle iv = v.val;
                    if (!v.is_i64) {
                        iv = emit_extsi_i32_to_i64(e, iv);
                    }
                    MLIR_ValueHandle pv = MLIR_CreateValueOpResult(
                        e->ctx, MLIR_INVALID_HANDLE, 0, e->ptr,
                        ssa_name(e), eloc(e, 0));
                    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->ptr;
                    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = pv;
                    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = iv;
                    emit_op(e, OP_TYPE_UNREGISTERED, str_lit("llvm.inttoptr"),
                            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
                    v.val = pv;
                    v.is_i64 = false;
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
                        // Build the function MLIR type for func.constant
                        // using the flat (sret/void-aware) signature so it
                        // matches the actual func.func definition.
                        MLIR_TypeHandle f_ty = MLIR_CreateTypeFunction(e->ctx,
                            fsig->flat_in_tys, fsig->n_flat_in,
                            fsig->flat_out_tys, fsig->n_flat_out);
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
            // Array-to-pointer decay: when an array variable appears as an
            // rvalue (e.g. passed to a pointer-typed function parameter), it
            // decays to a pointer to its first element. The alloca already
            // returns !llvm.ptr; just yield that address.
            if (ex->kind == EX_VAR) {
                Sym *s = lookup(e, sc, ex->name);
                if (s && (s->type.kind == TY_ARRAY_I32 ||
                          s->type.kind == TY_ARRAY_F32 ||
                          s->type.kind == TY_ARRAY_STRUCT ||
                          s->type.kind == TY_ARRAY_PTR_STRUCT ||
                          s->type.kind == TY_ARRAY_PTR_CHAR)) {
                    r.val = sym_addr(e, s);
                    r.is_ptr = true;
                    if (s->type.kind == TY_ARRAY_F32) r.ptr_elem = e->f32;
                    else if (s->type.kind == TY_ARRAY_PTR_STRUCT ||
                             s->type.kind == TY_ARRAY_PTR_CHAR) r.ptr_elem = e->ptr;
                    else if (s->type.kind == TY_ARRAY_I32 && s->type.array_elem_is_i64) r.ptr_elem = e->i64;
                    else if (s->type.kind == TY_ARRAY_I32 && s->type.array_elem_is_i8) r.ptr_elem = e->i8;
                    else r.ptr_elem = e->i32;
                    if (s->type.kind == TY_ARRAY_STRUCT) r.sdef = s->sdef;
                    return r;
                }
                // va_list value is the buffer pointer itself (the alloca
                // address). Passing va_list as an argument or to va_arg
                // wants this !llvm.ptr, not a load of its content.
                if (s && s->type.kind == TY_VA_LIST) {
                    r.val = sym_addr(e, s);
                    r.is_ptr = true;
                    return r;
                }
            }
            // Array-to-pointer decay for a struct field whose declared
            // type is an array (e.g. `struct S { uint8_t data[64]; };
            // ... s.data + i`). The rvalue is a pointer to the array's
            // first element; carrying the right ptr_elem lets downstream
            // pointer-arithmetic / indexing emit the correct stride.
            if (ex->kind == EX_FIELD) {
                SCtx parent = walk_struct_lhs(e, sc, ex->lhs);
                if (parent.ok) {
                    int fidx = struct_field_index(parent.sd, ex->name);
                    if (fidx >= 0) {
                        Type ft = parent.sd->fields.data[fidx].type;
                        if (ft.kind == TY_ARRAY_I32 ||
                            ft.kind == TY_ARRAY_F32 ||
                            ft.kind == TY_ARRAY_STRUCT ||
                            ft.kind == TY_ARRAY_PTR_STRUCT ||
                            ft.kind == TY_ARRAY_PTR_CHAR) {
                            sctx_push(e, &parent,
                                      parent.sd->is_union ? 0 : fidx);
                            MLIR_ValueHandle dyn[1];
                            size_t n_dyn = 0;
                            if (parent.dyn_index != MLIR_INVALID_HANDLE) {
                                dyn[0] = parent.dyn_index;
                                n_dyn = 1;
                            }
                            MLIR_ValueHandle p = emit_gep(e,
                                parent.base_ptr, parent.source_elem,
                                parent.const_path, parent.n_const_path,
                                dyn, n_dyn);
                            r.val = p;
                            r.is_ptr = true;
                            if (ft.kind == TY_ARRAY_F32) {
                                r.ptr_elem = e->f32;
                            } else if (ft.kind == TY_ARRAY_PTR_STRUCT ||
                                       ft.kind == TY_ARRAY_PTR_CHAR) {
                                r.ptr_elem = e->ptr;
                            } else if (ft.kind == TY_ARRAY_I32 &&
                                       ft.array_elem_is_i64) {
                                r.ptr_elem = e->i64;
                            } else if (ft.kind == TY_ARRAY_I32 &&
                                       ft.array_elem_is_i8) {
                                r.ptr_elem = e->i8;
                            } else {
                                r.ptr_elem = e->i32;
                            }
                            return r;
                        }
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
                        else if (s->type.kind == TY_PTR_I32)
                            v.ptr_elem = s->type.ptr_is_i64 ? e->i64
                                       : s->type.ptr_is_f32 ? e->f32
                                       : s->type.ptr_is_f64 ? e->f64
                                       : e->i32;
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
                            } else if (ft.kind == TY_PTR_CHAR) {
                                v.is_str = true;
                                v.ptr_elem = e->i8;
                            } else if (ft.kind == TY_PTR_I32) {
                                v.ptr_elem = ft.ptr_is_i64 ? e->i64 : e->i32;
                            } else if (ft.kind == TY_PTR_VOID) {
                                v.is_void_ptr = true;
                            }
                        }
                    }
                } else if (ex->kind == EX_INDEX) {
                    // Reading an element of a struct-field array of
                    // pointers: tag the result so downstream consumers
                    // (_tinyc_print(), -> chains) see the right element kind.
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
                                } else if (ft.kind == TY_PTR_PTR && ft.pointee) {
                                    // Indexing a struct field of type T**:
                                    // the loaded element is a single T*.
                                    // Tag with the inner type so '->'
                                    // chains and string ops work.
                                    Type *pe = ft.pointee;
                                    if (pe->kind == TY_PTR_CHAR) {
                                        v.is_str = true;
                                        v.ptr_elem = e->i8;
                                    } else if (pe->kind == TY_PTR_STRUCT) {
                                        v.sdef = find_struct(e, pe->struct_name);
                                    } else if (pe->kind == TY_PTR_I32) {
                                        v.ptr_elem = pe->ptr_is_i64 ? e->i64
                                                   : pe->ptr_is_f32 ? e->f32
                                                   : pe->ptr_is_f64 ? e->f64
                                                   : e->i32;
                                    } else if (pe->kind == TY_PTR_VOID) {
                                        v.is_void_ptr = true;
                                    } else if (pe->kind == TY_FNPTR) {
                                        Type *fnty = arena_new(e->arena, Type);
                                        *fnty = *pe;
                                        v.fnptr_ty = fnty;
                                    }
                                }
                            }
                        }
                    }
                    // Local variable indexed by a single subscript:
                    //   `a[i]` where `a` is `char **` / `char *arr[N]` /
                    //   `struct T *arr[N]`. The loaded element is a single
                    //   T*; without tagging `v.ptr_elem`, subsequent
                    //   pointer arithmetic (`a[i] + n`) falls back to the
                    //   `e->i32` default and GEPs with the wrong stride
                    //   (4 bytes per `n` instead of 1 byte per `n` for
                    //   char*) — observed during stage-3 self-host as the
                    //   driver.c `--export=NAME` parser advancing 36
                    //   bytes past the prefix instead of 9.
                    if (ex->lhs->kind == EX_VAR) {
                        Sym *vs = lookup(e, sc, ex->lhs->name);
                        if (vs) {
                            if (vs->type.kind == TY_PTR_PTR && vs->type.pointee) {
                                Type *pe = vs->type.pointee;
                                if (pe->kind == TY_PTR_CHAR) {
                                    v.is_str = true;
                                    v.ptr_elem = e->i8;
                                } else if (pe->kind == TY_PTR_STRUCT) {
                                    v.sdef = find_struct(e, pe->struct_name);
                                } else if (pe->kind == TY_PTR_I32) {
                                    v.ptr_elem = pe->ptr_is_i64 ? e->i64
                                               : pe->ptr_is_f32 ? e->f32
                                               : pe->ptr_is_f64 ? e->f64
                                               : e->i32;
                                } else if (pe->kind == TY_PTR_VOID) {
                                    v.is_void_ptr = true;
                                } else if (pe->kind == TY_FNPTR) {
                                    Type *fnty = arena_new(e->arena, Type);
                                    *fnty = *pe;
                                    v.fnptr_ty = fnty;
                                }
                            } else if (vs->type.kind == TY_ARRAY_PTR_CHAR) {
                                v.is_str = true;
                                v.ptr_elem = e->i8;
                            } else if (vs->type.kind == TY_ARRAY_PTR_STRUCT) {
                                v.sdef = find_struct(e, vs->type.struct_name);
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
                else if (s->type.kind == TY_I64) r.ptr_elem = e->i64;
                else if (s->type.kind == TY_F32) r.ptr_elem = e->f32;
                else if (s->type.kind == TY_F64) r.ptr_elem = e->f64;
                else if (s->type.kind == TY_PTR_I32 || s->type.kind == TY_PTR_CHAR ||
                         s->type.kind == TY_PTR_VOID || s->type.kind == TY_PTR_STRUCT ||
                         s->type.kind == TY_FNPTR || s->type.kind == TY_PTR_PTR) {
                    // &p where p is itself a pointer: yield a T** (storage
                    // !llvm.ptr) whose deref loads another !llvm.ptr.
                    r.ptr_elem = e->ptr;
                }
                else if (s->type.kind == TY_STRUCT) {
                    // &s where s is a struct value: the result is a real
                    // struct pointer. Carry the struct sdef + element type
                    // so downstream `(&s)->field` accesses (and the general
                    // `EX_DEREF` walk) can resolve the field chain.
                    r.sdef = s->sdef;
                    if (s->sdef) r.ptr_elem = find_struct_type(e, s->sdef);
                }
                return r;
            }
            if (ex->lhs->kind == EX_INDEX) {
                // &p->arr[i] / &(s.f)[i]: address of a struct element via
                // a chain. Use resolve_struct_source so the GEP descends
                // through the arrow/field chain instead of expecting a
                // simple-array base.
                Type lt = infer_expr_type(e, sc, ex->lhs);
                if (lt.kind == TY_STRUCT) {
                    StructDef *sd = find_struct(e, lt.struct_name);
                    if (sd) {
                        StructDef *src_sd = NULL;
                        MLIR_ValueHandle p = resolve_struct_source(
                            e, sc, ex->lhs, &src_sd);
                        if (p != MLIR_INVALID_HANDLE && src_sd == sd) {
                            r.val = p;
                            r.is_ptr = true;
                            r.sdef = sd;
                            return r;
                        }
                    }
                }
                // &arr[i] -> GEP address.
                LVal lv = emit_lvalue(e, sc, ex->lhs);
                r.val = lval_address(e, lv);
                r.is_ptr = true;
                r.ptr_elem = lv.elem_ty;
                return r;
            }
            // &<struct field>: take the address of any struct lvalue
            // (including struct-typed fields like `&buckets[i].value`).
            // For struct-valued lvalues we use walk_struct_lhs + GEP.
            if (ex->lhs->kind == EX_FIELD || ex->lhs->kind == EX_DEREF) {
                Type lt = infer_expr_type(e, sc, ex->lhs);
                if (lt.kind == TY_STRUCT) {
                    StructDef *sd = find_struct(e, lt.struct_name);
                    if (sd) {
                        StructDef *src_sd = NULL;
                        MLIR_ValueHandle p = resolve_struct_source(
                            e, sc, ex->lhs, &src_sd);
                        if (p != MLIR_INVALID_HANDLE && src_sd == sd) {
                            r.val = p;
                            r.is_ptr = true;
                            r.sdef = sd;
                            return r;
                        }
                    }
                }
                // Scalar field: fall back to emit_lvalue.
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
            // Struct-typed assignment: `lhs = rhs;` where both sides are
            // struct lvalues (e.g. `buckets[i].value = existing_value;`).
            // Resolve both addresses and emit a struct copy.
            {
                Type lt = infer_expr_type(e, sc, ex->lvalue);
                if (lt.kind == TY_STRUCT) {
                    StructDef *sd = find_struct(e, lt.struct_name);
                    if (sd) {
                        StructDef *src_sd = NULL;
                        StructDef *dst_sd = NULL;
                        MLIR_ValueHandle src = resolve_struct_source(
                            e, sc, ex->rhs_assign, &src_sd);
                        MLIR_ValueHandle dst = resolve_struct_source(
                            e, sc, ex->lvalue, &dst_sd);
                        if (src != MLIR_INVALID_HANDLE &&
                            dst != MLIR_INVALID_HANDLE &&
                            src_sd == sd && dst_sd == sd) {
                            emit_struct_copy(e, dst, src, sd);
                            r.is_ptr = true; r.sdef = sd; r.val = dst;
                            return r;
                        }
                    }
                }
            }
            LVal lv = emit_lvalue(e, sc, ex->lvalue);
            if (ex->is_post_step) {
                EVal old = load_lvalue(e, lv);
                bool sub = (ex->rhs_assign && ex->rhs_assign->op == OP_SUB);
                if (lv.elem_ty == e->ptr) {
                    // Pointer postfix `p++` / `p--`: step by sizeof(*p)
                    // via GEP, store new pointer, return the OLD pointer.
                    Type lt = infer_expr_type(e, sc, ex->lvalue);
                    MLIR_TypeHandle pe = e->i32;
                    if (lt.kind == TY_PTR_CHAR) pe = e->i8;
                    else if (lt.kind == TY_PTR_I32) {
                        if (lt.ptr_is_i64)      pe = e->i64;
                        else if (lt.ptr_is_f64) pe = e->f64;
                        else if (lt.ptr_is_f32) pe = e->f32;
                        else                    pe = e->i32;
                    }
                    else if (lt.kind == TY_PTR_PTR) pe = e->ptr;
                    int32_t step = sub ? -1 : 1;
                    MLIR_ValueHandle idx = emit_const_i32(e, step);
                    int32_t *path = arena_new_array(e->arena, int32_t, 1);
                    path[0] = LLVM_GEP_DYN;
                    MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                    dyn[0] = idx;
                    MLIR_ValueHandle stepped = emit_gep(e, old.val, pe, path, 1, dyn, 1);
                    store_lvalue(e, lv, stepped);
                    old.is_ptr = true;
                    old.ptr_elem = pe;
                    return old;
                }
                MLIR_TypeHandle ity = old.is_i64 ? e->i64 : e->i32;
                MLIR_ValueHandle one_v = old.is_i64 ? emit_const_i64(e, 1)
                                                    : emit_const_i32(e, 1);
                MLIR_OpType opty = sub ? OP_TYPE_ARITH_SUBI : OP_TYPE_ARITH_ADDI;
                string opname = sub ? str_lit("arith.subi") : str_lit("arith.addi");
                MLIR_ValueHandle stepped = emit_binop(e, opty, opname, ity,
                                                       old.val, one_v);
                MLIR_ValueHandle stored = coerce_eval(e,
                    (EVal){.val = stepped, .is_i64 = old.is_i64}, lv.elem_ty);
                store_lvalue(e, lv, stored);
                return old;
            }
            EVal v = emit_expr(e, sc, ex->rhs_assign);
            v.val = coerce_eval(e, v, lv.elem_ty);
            v.is_float = (lv.elem_ty == e->f32 || lv.elem_ty == e->f64);
            v.is_f64 = (lv.elem_ty == e->f64);
            v.is_i64 = (lv.elem_ty == e->i64);
            store_lvalue(e, lv, v.val);
            if (lv.elem_ty == e->i8) {
                // The stored value is i8, but the EVal we hand back to the
                // surrounding expression must be i32 (tinyc treats `char`
                // as `int` in arithmetic / comparisons). Match the load
                // path: zero-extend for `unsigned char`/`uint8_t`,
                // sign-extend otherwise.
                v.val = lv.is_unsigned ? emit_extui_i8_to_i32(e, v.val)
                                       : emit_extsi_i8_to_i32(e, v.val);
            }
            return v;
        }
        case EX_UN: {
            EVal a = emit_expr(e, sc, ex->lhs);
            if (ex->op == OP_NEG) {
                if (a.is_float) {
                    MLIR_ValueHandle z = a.is_f64 ? emit_const_f64(e, 0.0) : emit_const_f32(e, 0.0);
                    MLIR_TypeHandle ft = a.is_f64 ? e->f64 : e->f32;
                    r.val = emit_binop(e, OP_TYPE_ARITH_SUBF, str_lit("arith.subf"), ft, z, a.val);
                    r.is_float = true; r.is_f64 = a.is_f64;
                } else {
                    MLIR_ValueHandle z = a.is_i64 ? emit_const_i64(e, 0) : emit_const_i32(e, 0);
                    MLIR_TypeHandle ity = a.is_i64 ? e->i64 : e->i32;
                    r.val = emit_binop(e, OP_TYPE_ARITH_SUBI, str_lit("arith.subi"), ity, z, a.val);
                    r.is_i64 = a.is_i64;
                }
                return r;
            }
            if (ex->op == OP_BNOT) {
                if (a.is_float) {
                    EMIT_ERR(e, "~ not supported on floats");
                    r.val = emit_const_i32(e, 0);
                    return r;
                }
                if (a.is_i64) {
                    MLIR_ValueHandle ones = emit_const_i64(e, -1);
                    r.val = emit_binop(e, OP_TYPE_ARITH_XORI, str_lit("arith.xori"),
                                       e->i64, a.val, ones);
                    r.is_i64 = true;
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
                MLIR_ValueHandle cmp = emit_icmp_ptr(e, cmpi_pred_for(ex->op, false), a.val, b.val);
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
                MLIR_TypeHandle elem = (pv.ptr_elem != MLIR_INVALID_HANDLE) ? pv.ptr_elem
                    : (pv.sdef ? find_struct_type(e, pv.sdef) : e->i32);
                if (elem == MLIR_INVALID_HANDLE) elem = e->i32;
                MLIR_ValueHandle idx = iv.is_float ? emit_const_i32(e, 0) : iv.val;
                int32_t *path = arena_new_array(e->arena, int32_t, 1); path[0] = LLVM_GEP_DYN;
                MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, 1); dyn[0] = idx;
                r.val = emit_gep(e, pv.val, elem, path, 1, dyn, 1);
                r.is_ptr = true;
                r.ptr_elem = pv.ptr_elem;
                r.sdef = pv.sdef;
                r.is_str = pv.is_str;
                return r;
            }
            // Pointer minus integer: p - i -> GEP %p[-i].
            if (ex->op == OP_SUB && a.is_ptr && !b.is_ptr) {
                if (a.is_void_ptr) {
                    EMIT_ERR(e, "pointer arithmetic on 'void*' is not allowed");
                }
                MLIR_TypeHandle elem = (a.ptr_elem != MLIR_INVALID_HANDLE) ? a.ptr_elem
                    : (a.sdef ? find_struct_type(e, a.sdef) : e->i32);
                if (elem == MLIR_INVALID_HANDLE) elem = e->i32;
                MLIR_ValueHandle zero = emit_const_i32(e, 0);
                MLIR_ValueHandle neg = emit_binop(e, OP_TYPE_ARITH_SUBI,
                                                  str_lit("arith.subi"), e->i32, zero, b.val);
                int32_t *path = arena_new_array(e->arena, int32_t, 1); path[0] = LLVM_GEP_DYN;
                MLIR_ValueHandle *dyn = arena_new_array(e->arena, MLIR_ValueHandle, 1); dyn[0] = neg;
                r.val = emit_gep(e, a.val, elem, path, 1, dyn, 1);
                r.is_ptr = true;
                r.ptr_elem = a.ptr_elem;
                r.sdef = a.sdef;
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
                else if (elem == e->ptr) es = e->target_wasm32 ? 4 : 8;
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
            bool flt64 = a.is_f64;
            bool i64m = a.is_i64;
            bool unsgn = a.is_unsigned;
            MLIR_TypeHandle ity = i64m ? e->i64 : e->i32;
            MLIR_TypeHandle fty = flt64 ? e->f64 : e->f32;
            switch (ex->op) {
                case OP_ADD:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_ADDF, str_lit("arith.addf"), fty, a.val, b.val)
                                : emit_binop(e, OP_TYPE_ARITH_ADDI, str_lit("arith.addi"), ity, a.val, b.val);
                    r.is_float = flt; r.is_f64 = flt && flt64; r.is_i64 = i64m && !flt; r.is_unsigned = unsgn && !flt; return r;
                case OP_SUB:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_SUBF, str_lit("arith.subf"), fty, a.val, b.val)
                                : emit_binop(e, OP_TYPE_ARITH_SUBI, str_lit("arith.subi"), ity, a.val, b.val);
                    r.is_float = flt; r.is_f64 = flt && flt64; r.is_i64 = i64m && !flt; r.is_unsigned = unsgn && !flt; return r;
                case OP_MUL:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_MULF, str_lit("arith.mulf"), fty, a.val, b.val)
                                : emit_binop(e, OP_TYPE_ARITH_MULI, str_lit("arith.muli"), ity, a.val, b.val);
                    r.is_float = flt; r.is_f64 = flt && flt64; r.is_i64 = i64m && !flt; r.is_unsigned = unsgn && !flt; return r;
                case OP_DIV:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_DIVF, str_lit("arith.divf"), fty, a.val, b.val)
                                : (unsgn ? emit_binop(e, OP_TYPE_ARITH_DIVUI, str_lit("arith.divui"), ity, a.val, b.val)
                                         : emit_binop(e, OP_TYPE_ARITH_DIVSI, str_lit("arith.divsi"), ity, a.val, b.val));
                    r.is_float = flt; r.is_f64 = flt && flt64; r.is_i64 = i64m && !flt; r.is_unsigned = unsgn && !flt; return r;
                case OP_MOD:
                    if (flt) {
                        EMIT_ERR(e, "% not supported on floats");
                        r.val = emit_const_f32(e, 0.0); r.is_float = true; return r;
                    }
                    r.val = unsgn
                        ? emit_binop(e, OP_TYPE_ARITH_REMUI, str_lit("arith.remui"), ity, a.val, b.val)
                        : emit_binop(e, OP_TYPE_ARITH_REMSI, str_lit("arith.remsi"), ity, a.val, b.val);
                    r.is_i64 = i64m; r.is_unsigned = unsgn; return r;
                case OP_LT: case OP_LE: case OP_GT: case OP_GE:
                case OP_EQ: case OP_NE: {
                    MLIR_ValueHandle cmp = flt
                        ? emit_cmpf(e, cmpf_pred_for(ex->op), a.val, b.val)
                        : emit_cmpi(e, cmpi_pred_for(ex->op, unsgn), a.val, b.val);
                    r.val = bool_to_i32(e, cmp);
                    return r;
                }
                case OP_BAND:
                    if (flt) { EMIT_ERR(e, "& not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = emit_binop(e, OP_TYPE_ARITH_ANDI, str_lit("arith.andi"), ity, a.val, b.val);
                    r.is_i64 = i64m; r.is_unsigned = unsgn; return r;
                case OP_BOR:
                    if (flt) { EMIT_ERR(e, "| not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = emit_binop(e, OP_TYPE_ARITH_ORI, str_lit("arith.ori"), ity, a.val, b.val);
                    r.is_i64 = i64m; r.is_unsigned = unsgn; return r;
                case OP_BXOR:
                    if (flt) { EMIT_ERR(e, "^ not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = emit_binop(e, OP_TYPE_ARITH_XORI, str_lit("arith.xori"), ity, a.val, b.val);
                    r.is_i64 = i64m; r.is_unsigned = unsgn; return r;
                case OP_SHL:
                    if (flt) { EMIT_ERR(e, "<< not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = emit_binop(e, OP_TYPE_ARITH_SHLI, str_lit("arith.shli"), ity, a.val, b.val);
                    r.is_i64 = i64m; r.is_unsigned = unsgn; return r;
                case OP_SHR:
                    if (flt) { EMIT_ERR(e, ">> not supported on floats"); r.val = emit_const_i32(e, 0); return r; }
                    r.val = unsgn
                        ? emit_binop(e, OP_TYPE_ARITH_SHRUI, str_lit("arith.shrui"), ity, a.val, b.val)
                        : emit_binop(e, OP_TYPE_ARITH_SHRSI, str_lit("arith.shrsi"), ity, a.val, b.val);
                    r.is_i64 = i64m; r.is_unsigned = unsgn; return r;
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
                                 || str_eq(ex->callee, str_lit("va_copy"))
                                 || str_eq(ex->callee, str_lit("__builtin_va_start"))
                                 || str_eq(ex->callee, str_lit("__builtin_va_end"))
                                 || str_eq(ex->callee, str_lit("__builtin_va_copy")))) {
                bool is_start = str_eq(ex->callee, str_lit("va_start")) ||
                                str_eq(ex->callee, str_lit("__builtin_va_start"));
                bool is_copy  = str_eq(ex->callee, str_lit("va_copy")) ||
                                str_eq(ex->callee, str_lit("__builtin_va_copy"));
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
            // Built-in __builtin_sqrt / __builtin_sqrtf: lower to
            // llvm.intr.sqrt on f64 / f32 respectively.
            if (!indirect_fnty &&
                (str_eq(ex->callee, str_lit("__builtin_sqrt")) ||
                 str_eq(ex->callee, str_lit("__builtin_sqrtf")))) {
                bool is_f32 = str_eq(ex->callee, str_lit("__builtin_sqrtf"));
                if (ex->args.size != 1) {
                    EMIT_ERR(e, "{} expects 1 argument", ex->callee);
                    r.val = is_f32 ? emit_const_f32(e, 0.0) : emit_const_f64(e, 0.0);
                    r.is_float = true; r.is_f64 = !is_f32; return r;
                }
                EVal a = emit_expr(e, sc, ex->args.data[0]);
                MLIR_TypeHandle ft = is_f32 ? e->f32 : e->f64;
                if (!a.is_float) {
                    if (a.is_i64) a.val = emit_trunci_i64_to_i32(e, a.val);
                    a.val = is_f32 ? emit_sitofp(e, a.val) : emit_sitofp_f64(e, a.val);
                } else {
                    if (is_f32 && a.is_f64) {
                        // Truncate f64->f32 (rare here since arg should match).
                        // Fall back: just reuse value if compiler can't match.
                    } else if (!is_f32 && !a.is_f64) {
                        a.val = emit_fpext_f32_to_f64(e, a.val);
                    }
                }
                MLIR_ValueHandle res = MLIR_CreateValueOpResult(
                    e->ctx, MLIR_INVALID_HANDLE, 0, ft, ssa_name(e), eloc(e, 0));
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = ft;
                MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = res;
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = a.val;
                emit_op(e, OP_TYPE_UNREGISTERED, str_lit("llvm.intr.sqrt"),
                        rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
                r.val = res; r.is_float = true; r.is_f64 = !is_f32; return r;
            }
            // Built-in __builtin_unreachable(): marks an unreachable point
            // (e.g. after a noreturn syscall). tinyC does no flow analysis on
            // it, so treat it as a no-op — any code the C author placed after
            // it is already dead. Returns a dummy i32 for expression contexts.
            if (!indirect_fnty && str_eq(ex->callee, str_lit("__builtin_unreachable"))) {
                if (ex->args.size != 0) {
                    EMIT_ERR(e, "__builtin_unreachable expects 0 arguments");
                }
                r.val = emit_const_i32(e, 0); return r;
            }
            // Built-in __builtin_syscall6(num, a1, a2, a3, a4, a5, a6): a raw
            // OS syscall. Lowers to `func.call @__tinyc_syscall6` (a stub the
            // native llvm->aarch64 backend synthesises as a `svc` trap). All
            // seven operands and the result are `long` (i64). This is the
            // primitive corec's platform_*.c builds raw syscalls from; it has
            // meaning only for native targets (there are no raw syscalls on
            // wasm — use the WASI platform there).
            if (!indirect_fnty && str_eq(ex->callee, str_lit("__builtin_syscall6"))) {
                if (ex->args.size != 7) {
                    EMIT_ERR(e, "__builtin_syscall6 expects 7 arguments (num + 6)");
                    r.val = emit_const_i64(e, 0); r.is_i64 = true; return r;
                }
                e->need_syscall6_stub = true;
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 7);
                for (size_t k = 0; k < 7; k++) {
                    EVal av = emit_expr(e, sc, ex->args.data[k]);
                    ops[k] = av.is_ptr ? emit_ptrtoint_i64(e, av.val)
                                       : coerce_eval(e, av, e->i64);
                }
                MLIR_ValueHandle res = MLIR_CreateValueOpResult(
                    e->ctx, MLIR_INVALID_HANDLE, 0, e->i64, ssa_name(e), eloc(e, 0));
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i64;
                MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = res;
                MLIR_AttributeHandle ca = MLIR_CreateAttributeSymbolRef(
                    e->ctx, str_lit("callee"), str_lit("__tinyc_syscall6"));
                MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = ca;
                emit_op(e, OP_TYPE_FUNC_CALL, str_lit("func.call"),
                        rts, 1, rs, 1, ops, 7, as, 1, NULL, 0);
                r.val = res; r.is_i64 = true; return r;
            }
            // Built-in __builtin_wasm_memory_size(0) and
            // __builtin_wasm_memory_grow(0, n): lower to the
            // LLVM-style wasm intrinsics that the
            // llvm-dialect -> wasmssa pass recognizes. The first
            // argument (memory index) must be the integer literal 0;
            // we don't model multi-memory wasm yet so we just verify
            // and drop it. Both intrinsics produce i32 results.
            if (!indirect_fnty &&
                (str_eq(ex->callee, str_lit("__builtin_wasm_memory_size")) ||
                 str_eq(ex->callee, str_lit("__builtin_wasm_memory_grow")))) {
                bool is_grow = str_eq(ex->callee, str_lit("__builtin_wasm_memory_grow"));
                size_t want = is_grow ? 2 : 1;
                if (ex->args.size != want) {
                    EMIT_ERR(e, "{} expects {} argument(s)", ex->callee,
                             want == 1 ? str_lit("1") : str_lit("2"));
                    r.val = emit_const_i32(e, 0); return r;
                }
                // Argument 0 must be the immediate memory index. Only
                // 0 (default linear memory) is supported.
                Expr *aidx = ex->args.data[0];
                if (aidx->kind != EX_INT || aidx->int_value != 0) {
                    EMIT_ERR(e, "{}: first arg must be the constant 0",
                             ex->callee);
                }
                MLIR_TypeHandle rty = e->i32;
                MLIR_ValueHandle res = MLIR_CreateValueOpResult(
                    e->ctx, MLIR_INVALID_HANDLE, 0, rty, ssa_name(e), eloc(e, 0));
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = rty;
                MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = res;
                if (is_grow) {
                    EVal pages = emit_expr(e, sc, ex->args.data[1]);
                    MLIR_ValueHandle p32 = coerce_eval(e, pages, e->i32);
                    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = p32;
                    emit_op(e, OP_TYPE_UNREGISTERED, str_lit("llvm.intr.wasm.memory.grow"),
                            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
                } else {
                    emit_op(e, OP_TYPE_UNREGISTERED, str_lit("llvm.intr.wasm.memory.size"),
                            rts, 1, rs, 1, NULL, 0, NULL, 0, NULL, 0);
                }
                r.val = res; return r;
            }
            // Built-in malloc(size) -> !llvm.ptr. The libc signature
            // takes i64, so accept any integer expression (i32, i64,
            // pointer-sized) and coerce up to i64. Using emit_expr +
            // coerce_eval avoids emitting a no-op `arith.extsi i64->i64`
            // when the source is already size_t-sized (e.g. `size_t n;
            // malloc(n)` on a 64-bit host).
            if (!indirect_fnty && str_eq(ex->callee, str_lit("malloc"))) {
                // If the user provided their own `extern void *malloc(...);`
                // prototype, mark that signature as referenced so the
                // forward-decl loop in emit_program emits a proper
                // `func.func` declaration for it. The special-case path
                // here bypasses find_sig (which is what normally tracks
                // usage), so without this we'd drop the user decl AND
                // skip our auto-decl (because have_user_malloc is true),
                // leaving the @malloc call unresolved.
                (void)find_sig(e, ex->callee);
                if (ex->args.size != 1) {
                    EMIT_ERR(e, "malloc expects 1 argument");
                    r.val = emit_null_ptr(e); r.is_ptr = true; return r;
                }
                EVal sz_v = emit_expr(e, sc, ex->args.data[0]);
                MLIR_ValueHandle sz_coerced = coerce_eval(e, sz_v, e->size_t_ty);
                MLIR_ValueHandle res = MLIR_CreateValueOpResult(
                    e->ctx, MLIR_INVALID_HANDLE, 0, e->ptr, ssa_name(e), eloc(e, 0));
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->ptr;
                MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = res;
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = sz_coerced;
                MLIR_AttributeHandle ca = MLIR_CreateAttributeSymbolRef(
                    e->ctx, str_lit("callee"), str_lit("malloc"));
                MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = ca;
                emit_op(e, OP_TYPE_FUNC_CALL, str_lit("func.call"),
                        rts, 1, rs, 1, ops, 1, as, 1, NULL, 0);
                r.val = res; r.is_ptr = true; return r;
            }
            if (!indirect_fnty && str_eq(ex->callee, str_lit("free"))) {
                (void)find_sig(e, ex->callee);
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
                // Struct-returning indirect call uses sret ABI: a hidden
                // first !llvm.ptr param receives the buffer the callee
                // writes the struct into; the call itself returns void.
                bool ret_is_struct = (fnty->fnptr_ret->kind == TY_STRUCT);
                StructDef *ret_sd = NULL;
                MLIR_TypeHandle ret_st_ty = MLIR_INVALID_HANDLE;
                MLIR_ValueHandle sret_buf = MLIR_INVALID_HANDLE;
                if (ret_is_struct) {
                    ret_sd = find_struct(e, fnty->fnptr_ret->struct_name);
                    if (!ret_sd) {
                        EMIT_ERR(e, "unknown struct return type {}",
                                 fnty->fnptr_ret->struct_name);
                        r.val = emit_const_i32(e, 0);
                        return r;
                    }
                    ret_st_ty = find_struct_type(e, ret_sd);
                    sret_buf = emit_alloca(e, ret_st_ty);
                }
                // Build the function MLIR type that func.call_indirect needs.
                size_t np = (size_t)fnty->fnptr_nparams;
                size_t n_in_tys = np + (ret_is_struct ? 1 : 0);
                MLIR_TypeHandle *in_tys = arena_new_array(e->arena, MLIR_TypeHandle,
                                                           n_in_tys ? n_in_tys : 1);
                size_t in_off = 0;
                if (ret_is_struct) in_tys[in_off++] = e->ptr;
                for (size_t i = 0; i < np; i++) {
                    if (fnty->fnptr_params[i].kind == TY_STRUCT ||
                        fnty->fnptr_params[i].kind == TY_PTR_STRUCT)
                        in_tys[in_off++] = e->ptr;
                    else
                        in_tys[in_off++] = scalar_mlir_type(e, fnty->fnptr_params[i].kind);
                }
                MLIR_TypeHandle rty = ret_is_struct
                    ? e->i32 /* unused */
                    : scalar_mlir_type(e, fnty->fnptr_ret->kind);
                MLIR_TypeHandle out_tys[1] = { rty };
                size_t n_out_tys = ret_is_struct ? 0 : 1;
                MLIR_TypeHandle f_ty = MLIR_CreateTypeFunction(e->ctx,
                    in_tys, n_in_tys, out_tys, n_out_tys);
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

                size_t n_ops = 1 + na + (ret_is_struct ? 1 : 0);
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle,
                                                         n_ops ? n_ops : 1);
                size_t op_off = 0;
                ops[op_off++] = callee_v;
                if (ret_is_struct) ops[op_off++] = sret_buf;
                for (size_t i = 0; i < na; i++) {
                    Type pty = (i < (size_t)fnty->fnptr_nparams)
                        ? fnty->fnptr_params[i] : (Type){0};
                    if (pty.kind == TY_STRUCT) {
                        StructDef *psd = find_struct(e, pty.struct_name);
                        StructDef *asd = NULL;
                        MLIR_ValueHandle src = resolve_struct_source(e, sc,
                                ex->args.data[i], &asd);
                        if (src == MLIR_INVALID_HANDLE || (psd && asd != psd)) {
                            EMIT_ERR(e, "indirect call struct arg type mismatch");
                            ops[op_off++] = src;
                            continue;
                        }
                        MLIR_TypeHandle st = find_struct_type(e, psd);
                        MLIR_ValueHandle tmp = emit_alloca(e, st);
                        emit_struct_copy(e, tmp, src, psd);
                        ops[op_off++] = tmp;
                    } else if (pty.kind == TY_PTR_STRUCT) {
                        StructDef *asd = NULL;
                        MLIR_ValueHandle src = resolve_struct_source(e, sc,
                                ex->args.data[i], &asd);
                        ops[op_off++] = src;
                    } else {
                        EVal av = emit_expr(e, sc, ex->args.data[i]);
                        TypeKind want = (i < (size_t)fnty->fnptr_nparams)
                            ? fnty->fnptr_params[i].kind : TY_I32;
                        ops[op_off++] = coerce_eval(e, av, scalar_mlir_type(e, want));
                    }
                }
                size_t n_rts = ret_is_struct ? 0 : 1;
                MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1);
                MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1);
                if (!ret_is_struct) {
                    rts[0] = rty;
                    rs[0] = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                      rty, ssa_name(e), eloc(e, 0));
                }
                emit_op(e, OP_TYPE_FUNC_CALL_INDIRECT, str_lit("func.call_indirect"),
                        rts, n_rts, rs, n_rts, ops, n_ops, NULL, 0, NULL, 0);
                if (ret_is_struct) {
                    r.val = sret_buf;
                    r.is_ptr = true;
                    r.sdef = ret_sd;
                    return r;
                }
                r.val = rs[0];
                r.is_float = (fnty->fnptr_ret->kind == TY_F32 || fnty->fnptr_ret->kind == TY_F64);
                r.is_f64 = (fnty->fnptr_ret->kind == TY_F64);
                r.is_i64 = (fnty->fnptr_ret->kind == TY_I64);
                r.is_unsigned = fnty->fnptr_ret->int_unsigned;
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
            r.is_float = (sig->ret.type.kind == TY_F32 || sig->ret.type.kind == TY_F64);
            r.is_f64 = (sig->ret.type.kind == TY_F64);
            r.is_i64 = (sig->ret.type.kind == TY_I64);
            r.is_unsigned = sig->ret.type.int_unsigned;
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
            if (t.kind == TY_ARRAY_STRUCT) {
                // Array-of-struct compound literal `(S[]){{...}, {...}, ...}`.
                // Each arg is itself an EX_COMPOUND for a single struct
                // element. Length defaults to the number of args when the
                // bracket was empty (`(S[]){...}`).
                StructDef *sd = find_struct(e, t.struct_name);
                if (!sd) { EMIT_ERR(e, "unknown struct in array compound literal"); return r; }
                int64_t n = (int64_t)ex->args.size;
                if (t.array_len > 0) {
                    if (t.array_len < n) n = t.array_len;
                } else {
                    t.array_len = n;
                }
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                MLIR_TypeHandle arr_ty = MLIR_CreateTypeLLVMArray(
                    e->ctx, st_ty, (uint64_t)t.array_len);
                MLIR_ValueHandle addr = emit_alloca(e, arr_ty);
                for (int64_t i = 0; i < n; i++) {
                    Expr *elem = ex->args.data[i];
                    int32_t path[2] = {0, (int32_t)i};
                    MLIR_ValueHandle p = emit_gep(e, addr, arr_ty, path, 2, NULL, 0);
                    if (elem->kind == EX_COMPOUND) {
                        // Fill element struct fields directly (avoid
                        // emitting a second alloca + memcpy).
                        size_t fn = elem->args.size;
                        if (fn > sd->fields.size) fn = sd->fields.size;
                        // Zero first.
                        int32_t pz[1] = {0};
                        emit_struct_zero(e, p, st_ty, sd, pz, 1);
                        for (size_t fi = 0; fi < fn; fi++) {
                            Type ft = sd->fields.data[fi].type;
                            int32_t fpath[2] = {0, (int32_t)fi};
                            MLIR_ValueHandle fp = emit_gep(e, p, st_ty, fpath, 2, NULL, 0);
                            EVal vv = emit_expr(e, sc, elem->args.data[fi]);
                            MLIR_TypeHandle want = scalar_mlir_type(e, ft.kind);
                            MLIR_ValueHandle sv;
                            if (ft.kind == TY_PTR_I32 || ft.kind == TY_PTR_CHAR ||
                                ft.kind == TY_PTR_VOID || ft.kind == TY_PTR_STRUCT ||
                                ft.kind == TY_FNPTR || ft.kind == TY_PTR_PTR) {
                                sv = vv.val;
                            } else {
                                sv = coerce_eval(e, vv, want);
                            }
                            emit_store_v(e, sv, fp);
                        }
                    } else {
                        // Allow a struct-typed expression as an element
                        // (e.g. an existing struct local) by copying.
                        StructDef *esd = NULL;
                        MLIR_ValueHandle src = resolve_struct_source(e, sc, elem, &esd);
                        if (src != MLIR_INVALID_HANDLE && esd == sd) {
                            emit_struct_copy(e, p, src, sd);
                        } else {
                            EMIT_ERR(e, "array compound literal element type mismatch");
                        }
                    }
                }
                r.val = addr;
                r.is_ptr = true;
                r.sdef = sd;
                return r;
            }
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
                    size_t fi = i;
                    if (ex->compound_field_names &&
                        ex->compound_field_names[i].size > 0) {
                        bool found = false;
                        for (size_t k = 0; k < sd->fields.size; k++) {
                            if (str_eq(sd->fields.data[k].name,
                                       ex->compound_field_names[i])) {
                                fi = k; found = true; break;
                            }
                        }
                        if (!found) {
                            EMIT_ERR(e, "unknown field '{}' in designated initializer",
                                     ex->compound_field_names[i]);
                            continue;
                        }
                    }
                    Type ft = sd->fields.data[fi].type;
                    int32_t path[2] = {0, (int32_t)fi};
                    MLIR_ValueHandle p =
                        emit_gep(e, addr, st_ty, path, 2, NULL, 0);
                    if (ft.kind == TY_STRUCT) {
                        StructDef *fsd = find_struct(e, ft.struct_name);
                        if (!fsd) {
                            EMIT_ERR(e, "unknown struct type for compound-literal field");
                            continue;
                        }
                        Expr *ae = ex->args.data[i];
                        if (ae->kind == EX_INT && ae->int_value == 0) {
                            // Already zero-initialized by emit_struct_zero above.
                            continue;
                        }
                        // Nested aggregate `{ ... }` with no explicit
                        // cast_type: propagate the surrounding field's
                        // struct type so emit_expr resolves it.
                        if (ae->kind == EX_COMPOUND &&
                            ae->cast_type.kind == TY_VOID) {
                            ae->cast_type = ft;
                        }
                        StructDef *src_sd = NULL;
                        MLIR_ValueHandle src = resolve_struct_source(e, sc, ae, &src_sd);
                        if (src != MLIR_INVALID_HANDLE && src_sd == fsd) {
                            emit_struct_copy(e, p, src, fsd);
                        } else {
                            EVal vv = emit_expr(e, sc, ae);
                            if (vv.is_ptr && vv.val != MLIR_INVALID_HANDLE) {
                                emit_struct_copy(e, p, vv.val, fsd);
                            } else {
                                EMIT_ERR(e, "compound-literal struct field needs a struct value");
                            }
                        }
                        continue;
                    }
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
    if (sd->is_union) {
        // The union's llvm body is a single blob field (index 0) of
        // `count` alignment-wide integers. Zero each blob element.
        MLIR_TypeHandle uelem; int64_t ues, uc;
        union_blob_layout(e, sd, &uelem, &ues, &uc);
        MLIR_ValueHandle z = (ues == 8) ? emit_const_i64(e, 0)
                           : (ues == 1) ? emit_const_i8(e, 0)
                                        : emit_const_i32(e, 0);
        for (int64_t j = 0; j < uc; j++) {
            size_t total = n_prefix + 2;
            int32_t *p2 = arena_new_array(e->arena, int32_t, total);
            for (size_t k = 0; k < n_prefix; k++) p2[k] = prefix[k];
            p2[n_prefix] = 0;
            p2[n_prefix + 1] = (int32_t)j;
            MLIR_ValueHandle p = emit_gep(e, base, source_elem, p2, total, NULL, 0);
            emit_store_v(e, z, p);
        }
        return;
    }
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
                   ft.kind == TY_PTR_CHAR || ft.kind == TY_PTR_VOID ||
                   ft.kind == TY_PTR_PTR || ft.kind == TY_FNPTR) {
            MLIR_ValueHandle p = emit_gep(e, base, source_elem, path, n_path, NULL, 0);
            emit_store_v(e, emit_null_ptr(e), p);
        } else if (ft.kind == TY_I64) {
            MLIR_ValueHandle p = emit_gep(e, base, source_elem, path, n_path, NULL, 0);
            emit_store_v(e, emit_const_i64(e, 0), p);
        } else if (ft.kind == TY_F64) {
            MLIR_ValueHandle p = emit_gep(e, base, source_elem, path, n_path, NULL, 0);
            emit_store_v(e, emit_const_f64(e, 0.0), p);
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
            } else if (ft.array_elem_is_i64) { z = emit_const_i64(e, 0); }
            else if (ft.array_elem_is_i8) { z = emit_const_i8(e, 0); }
            else { z = emit_const_i32(e, 0); }
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

// Emit `ex` for its side effects only — the result value is discarded.
// Handles a few "void-typed" expression forms (void calls, `(void)X`,
// and `cond ? a : b` where both arms are themselves statement-level)
// without trying to compute a result value; falls back to plain
// `emit_expr` for the general case. This is what makes idioms like
//     ((cond) ? (void)0 : abort())   // as produced by assert()
// compile cleanly: emit_expr would otherwise error on the void-returning
// arm "void-returning call has no value (use as a statement)".
static bool expr_is_void_statement(E *e, Expr *ex) {
    if (!ex) return false;
    if (ex->kind == EX_CALL) {
        FuncSig *sig = find_sig(e, ex->callee);
        return sig && (sig->ret.type.kind == TY_VOID ||
                       sig->ret.type.kind == TY_STRUCT);
    }
    if (ex->kind == EX_CAST && ex->cast_type.kind == TY_VOID) return true;
    if (ex->kind == EX_TERNARY) {
        return expr_is_void_statement(e, ex->rhs) &&
               expr_is_void_statement(e, ex->lvalue);
    }
    return false;
}

static void emit_expr_for_side_effect(E *e, Scope *sc, Expr *ex) {
    if (e->terminated) return;
    if (ex->kind == EX_CALL) {
        FuncSig *sig = find_sig(e, ex->callee);
        if (sig && sig->ret.type.kind == TY_STRUCT) {
            emit_flat_call(e, sc, sig, ex->args, NULL, MLIR_INVALID_HANDLE);
            return;
        }
        if (sig && sig->ret.type.kind == TY_VOID) {
            emit_flat_call(e, sc, sig, ex->args, NULL, MLIR_INVALID_HANDLE);
            return;
        }
    }
    if (ex->kind == EX_CAST && ex->cast_type.kind == TY_VOID) {
        emit_expr_for_side_effect(e, sc, ex->lhs);
        return;
    }
    if (ex->kind == EX_TERNARY &&
        expr_is_void_statement(e, ex->rhs) &&
        expr_is_void_statement(e, ex->lvalue)) {
        // `c ? a : b` as a statement where both arms are void: lower as
        // an if/else branch without an alloca/load for the result.
        EVal cv = emit_expr(e, sc, ex->lhs);
        MLIR_ValueHandle cb = emit_to_bool_i1(e, cv);
        MLIR_BlockHandle then_blk = new_cfg_block(e);
        MLIR_BlockHandle else_blk = new_cfg_block(e);
        MLIR_BlockHandle merge_blk = new_cfg_block(e);
        emit_cond_branch(e, cb, then_blk, else_blk);

        e->cur_block = then_blk; e->terminated = false;
        emit_expr_for_side_effect(e, sc, ex->rhs);
        if (!e->terminated) emit_branch(e, merge_blk);

        e->cur_block = else_blk; e->terminated = false;
        emit_expr_for_side_effect(e, sc, ex->lvalue);
        if (!e->terminated) emit_branch(e, merge_blk);

        e->cur_block = merge_blk; e->terminated = false;
        return;
    }
    (void)emit_expr(e, sc, ex);
}

static void emit_stmt(E *e, Scope *sc, Stmt *st) {
    if (e->terminated && st->kind != ST_LABEL) return;
    e->cur_line = st->line;
    switch (st->kind) {
        case ST_EXPR: {
            emit_expr_for_side_effect(e, sc, st->expr);
            return;
        }
        case ST_DECL: {
            Sym *sy = arena_new(e->arena, Sym);
            *sy = (Sym){0};
            sy->name = st->decl_name;
            sy->type = st->decl_type;

            // Function-local `static`: the parser has already registered
            // a module-scope global for this declaration. Bind the local
            // name to that global; do not allocate a stack slot and do
            // not re-emit the initializer (the Global captures it).
            if (st->decl_static_global_sym.size > 0) {
                sy->is_global = true;
                sy->global_sym = st->decl_static_global_sym;
                sy->next = sc->head;
                sc->head = sy;
                return;
            }

            // Unsized `T arr[] = {a, b, ...}`: infer length from the
            // aggregate initializer's argument count.
            if ((st->decl_type.kind == TY_ARRAY_I32 ||
                 st->decl_type.kind == TY_ARRAY_PTR_CHAR ||
                 st->decl_type.kind == TY_ARRAY_STRUCT) &&
                st->decl_type.array_len == 0 &&
                !st->decl_type.array_len_expr &&
                st->decl_init && st->decl_init->kind == EX_COMPOUND) {
                int64_t n = (int64_t)st->decl_init->args.size;
                if (n <= 0) {
                    EMIT_ERR(e, "unsized array '{}[]' requires a non-empty initializer", st->decl_name);
                    n = 1;
                }
                st->decl_type.array_len = n;
                sy->type.array_len = n;
            }
            // Constant-expression array length, e.g. `T arr[sizeof(x)/sizeof(x[0])]`.
            // Fold against the current scope so sizeof of in-scope locals works.
            resolve_array_len(e, sc, &st->decl_type);
            sy->type.array_len = st->decl_type.array_len;

            // C semantics: an identifier's scope begins immediately after its
            // declarator, so it is visible inside its own initializer (e.g.
            // `T *p = alloc(sizeof(*p))`). Register the symbol now, before the
            // initializer is emitted, so type/size queries on it resolve
            // correctly. (The static-global case above has already returned.)
            sy->next = sc->head;
            sc->head = sy;

            if (st->decl_type.kind == TY_VA_LIST) {
                // Allocate a 32-byte buffer (sufficient for x86_64-SysV
                // and aarch64 va_list layouts on Linux/macOS/Windows). We
                // hand out the pointer; va_start/va_arg/va_end consume it.
                MLIR_TypeHandle buf_ty = MLIR_CreateTypeLLVMArray(e->ctx, e->i8, 32);
                sy->addr = emit_alloca(e, buf_ty);
                break;
            }
            if (st->decl_type.kind == TY_PTR_I32 || st->decl_type.kind == TY_PTR_VOID ||
                st->decl_type.kind == TY_PTR_PTR) {
                sy->addr = emit_alloca(e, e->ptr);
                if (st->decl_init) {
                    EVal iv = emit_expr(e, sc, st->decl_init);
                    if (iv.is_ptr) {
                        emit_store_v(e, iv.val, sy->addr);
                    } else if (st->decl_init->kind == EX_INT && st->decl_init->int_value == 0) {
                        // Null-pointer constant: `T *p = 0;` / `T *p = NULL;`.
                        emit_store_v(e, emit_null_ptr(e), sy->addr);
                    } else {
                        EMIT_ERR(e, "pointer initializer must be a pointer expression");
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
                    if (st->decl_init->kind == EX_ADDR &&
                        st->decl_init->lhs->kind == EX_VAR) {
                        Sym *target = lookup(e, sc, st->decl_init->lhs->name);
                        if (!target || target->type.kind != TY_STRUCT || target->sdef != sd) {
                            EMIT_ERR(e, "struct* target type mismatch");
                        } else {
                            emit_store_v(e, sym_addr(e, target), sy->addr);
                        }
                    } else {
                        // Allow any expression yielding a pointer (e.g., a
                        // function call returning struct*, or a cast).
                        EVal iv = emit_expr(e, sc, st->decl_init);
                        MLIR_ValueHandle p = iv.is_ptr ? iv.val : emit_null_ptr(e);
                        emit_store_v(e, p, sy->addr);
                    }
                }
            } else if (st->decl_type.kind == TY_ARRAY_I32) {
                MLIR_TypeHandle elem_ty = st->decl_type.array_elem_is_i64 ? e->i64
                                        : st->decl_type.array_elem_is_i8  ? e->i8
                                        : e->i32;
                MLIR_TypeHandle arr_ty;
                if (st->decl_type.array_len2 != 0) {
                    MLIR_TypeHandle inner = MLIR_CreateTypeLLVMArray(
                        e->ctx, elem_ty, (uint64_t)st->decl_type.array_len2);
                    arr_ty = MLIR_CreateTypeLLVMArray(
                        e->ctx, inner, (uint64_t)st->decl_type.array_len);
                } else {
                    arr_ty = MLIR_CreateTypeLLVMArray(
                        e->ctx, elem_ty, (uint64_t)st->decl_type.array_len);
                }
                sy->addr = emit_alloca(e, arr_ty);
                if (st->decl_init) {
                    // `char arr[N] = "string literal"` — copy the string
                    // bytes (lexer already includes a trailing NUL) into
                    // the local buffer, zero-padding any remaining bytes.
                    if (st->decl_init->kind == EX_STR &&
                        st->decl_type.array_elem_is_i8 &&
                        st->decl_type.array_len2 == 0) {
                        string lit = st->decl_init->name;
                        int64_t alen = st->decl_type.array_len;
                        if ((int64_t)lit.size > alen) {
                            EMIT_ERR(e, "string initializer ({} bytes) too long for char[{}]",
                                     (int64_t)lit.size, alen);
                        }
                        int64_t lim = (int64_t)lit.size < alen ? (int64_t)lit.size : alen;
                        for (int64_t k = 0; k < lim; k++) {
                            MLIR_ValueHandle bv = emit_const_i8(e, (int64_t)(uint8_t)lit.str[k]);
                            int32_t path[2] = {0, (int32_t)k};
                            MLIR_ValueHandle p = emit_gep(e, sy->addr, arr_ty, path, 2, NULL, 0);
                            emit_store_v(e, bv, p);
                        }
                        if (lim < alen) {
                            MLIR_ValueHandle z8 = emit_const_i8(e, 0);
                            for (int64_t k = lim; k < alen; k++) {
                                int32_t path[2] = {0, (int32_t)k};
                                MLIR_ValueHandle p = emit_gep(e, sy->addr, arr_ty, path, 2, NULL, 0);
                                emit_store_v(e, z8, p);
                            }
                        }
                        return;
                    }
                    // Support `= {0}` (zero-initialize) by zeroing every
                    // element. Also support a positional list `{v0,v1,...}`
                    // for 1-D arrays where each value is an int expression.
                    bool is_zero_init = false;
                    bool is_list_init = false;
                    if (st->decl_init->kind == EX_COMPOUND) {
                        size_t n = st->decl_init->args.size;
                        is_zero_init = (n == 0);
                        if (n == 1) {
                            Expr *a = st->decl_init->args.data[0];
                            if (a->kind == EX_INT && a->int_value == 0) {
                                is_zero_init = true;
                            }
                        }
                        if (!is_zero_init && st->decl_type.array_len2 == 0) {
                            is_list_init = true;
                        }
                    }
                    if (!is_zero_init && !is_list_init) {
                        EMIT_ERR(e, "array initializers are not supported");
                    } else if (is_list_init) {
                        size_t n = st->decl_init->args.size;
                        if ((int64_t)n != st->decl_type.array_len) {
                            EMIT_ERR(e, "int array initializer count {} does not match length {}",
                                     (int64_t)n, st->decl_type.array_len);
                        }
                        int64_t lim = (int64_t)n < st->decl_type.array_len ? (int64_t)n : st->decl_type.array_len;
                        bool elem_i8  = st->decl_type.array_elem_is_i8;
                        bool elem_i64 = st->decl_type.array_elem_is_i64;
                        for (int64_t a = 0; a < lim; a++) {
                            MLIR_ValueHandle iv = emit_expr_i32(e, sc, st->decl_init->args.data[a]);
                            // The element load/store width must match the
                            // declared element type. emit_expr_i32 yields i32;
                            // truncate to i8 (or extend to i64) so the store
                            // doesn't overflow the slot and clobber adjacent
                            // memory (which on wasm32 can corrupt the
                            // immediately-following rodata page).
                            if (elem_i8)       iv = emit_trunci_to_i8(e, iv);
                            else if (elem_i64) iv = emit_extsi_i32_to_i64(e, iv);
                            int32_t path[2] = {0, (int32_t)a};
                            MLIR_ValueHandle p = emit_gep(e, sy->addr, arr_ty, path, 2, NULL, 0);
                            emit_store_v(e, iv, p);
                        }
                    } else {
                        int64_t n1 = st->decl_type.array_len;
                        int64_t n2 = st->decl_type.array_len2 ? st->decl_type.array_len2 : 1;
                        bool is_2d = (st->decl_type.array_len2 != 0);
                        // Use a zero value whose width matches the element
                        // type so the i32 store doesn't overflow narrower
                        // (i8) slots.
                        MLIR_ValueHandle z = st->decl_type.array_elem_is_i8
                            ? emit_const_i8(e, 0)
                            : st->decl_type.array_elem_is_i64
                                ? emit_const_i64(e, 0)
                                : emit_const_i32(e, 0);
                        for (int64_t a = 0; a < n1; a++) {
                            for (int64_t b = 0; b < n2; b++) {
                                int32_t path[3];
                                size_t np = 0;
                                path[np++] = 0;
                                path[np++] = (int32_t)a;
                                if (is_2d) path[np++] = (int32_t)b;
                                MLIR_ValueHandle p = emit_gep(
                                    e, sy->addr, arr_ty, path, np, NULL, 0);
                                emit_store_v(e, z, p);
                            }
                        }
                    }
                }
            } else if (st->decl_type.kind == TY_ARRAY_PTR_CHAR) {
                MLIR_TypeHandle arr_ty = MLIR_CreateTypeLLVMArray(
                    e->ctx, e->ptr, (uint64_t)st->decl_type.array_len);
                sy->addr = emit_alloca(e, arr_ty);
                if (st->decl_init) {
                    if (st->decl_init->kind != EX_COMPOUND) {
                        EMIT_ERR(e, "char* array initializer must be an aggregate '{...}'");
                    } else {
                        size_t n = st->decl_init->args.size;
                        // `= {0}` zero-init shorthand.
                        bool is_zero = (n == 0) ||
                                       (n == 1 &&
                                        st->decl_init->args.data[0]->kind == EX_INT &&
                                        st->decl_init->args.data[0]->int_value == 0);
                        if (is_zero) {
                            MLIR_ValueHandle nullp = emit_null_ptr(e);
                            for (int64_t a = 0; a < st->decl_type.array_len; a++) {
                                int32_t path[2] = {0, (int32_t)a};
                                MLIR_ValueHandle p = emit_gep(e, sy->addr, arr_ty, path, 2, NULL, 0);
                                emit_store_v(e, nullp, p);
                            }
                        } else {
                            if ((int64_t)n != st->decl_type.array_len) {
                                EMIT_ERR(e, "char* array initializer count {} does not match length {}",
                                         (int64_t)n, st->decl_type.array_len);
                            }
                            int64_t lim = (int64_t)n < st->decl_type.array_len ? (int64_t)n : st->decl_type.array_len;
                            for (int64_t a = 0; a < lim; a++) {
                                EVal iv = emit_expr(e, sc, st->decl_init->args.data[a]);
                                MLIR_ValueHandle p_val = iv.is_ptr ? iv.val : emit_null_ptr(e);
                                int32_t path[2] = {0, (int32_t)a};
                                MLIR_ValueHandle p = emit_gep(e, sy->addr, arr_ty, path, 2, NULL, 0);
                                emit_store_v(e, p_val, p);
                            }
                        }
                    }
                }
            } else if (st->decl_type.kind == TY_ARRAY_STRUCT) {
                StructDef *sd = find_struct(e, st->decl_type.struct_name);
                if (!sd) {
                    EMIT_ERR(e, "unknown struct type {}", st->decl_type.struct_name);
                    return;
                }
                sy->sdef = sd;
                MLIR_TypeHandle st_ty = find_struct_type(e, sd);
                // Allow `S name[] = { ... };` — infer length from the
                // initializer count when no explicit dimension was given.
                int64_t alen = st->decl_type.array_len;
                if (st->decl_init &&
                    st->decl_init->kind == EX_COMPOUND &&
                    alen == 0 && !st->decl_type.array_len_expr) {
                    alen = (int64_t)st->decl_init->args.size;
                    st->decl_type.array_len = alen;
                    sy->type.array_len = alen;
                }
                MLIR_TypeHandle arr_ty = MLIR_CreateTypeLLVMArray(
                    e->ctx, st_ty, (uint64_t)alen);
                sy->addr = emit_alloca(e, arr_ty);
                if (st->decl_init) {
                    if (st->decl_init->kind != EX_COMPOUND) {
                        EMIT_ERR(e, "array-of-struct initializer must be a brace list");
                    } else {
                        size_t n = st->decl_init->args.size;
                        if ((int64_t)n > alen) {
                            EMIT_ERR(e, "too many array-of-struct initializers");
                            n = (size_t)alen;
                        }
                        for (size_t k = 0; k < n; k++) {
                            Expr *elem = st->decl_init->args.data[k];
                            int32_t epath[2] = {0, (int32_t)k};
                            MLIR_ValueHandle pelem = emit_gep(
                                e, sy->addr, arr_ty, epath, 2, NULL, 0);
                            // Zero this slot first so any unspecified
                            // fields read back as 0 / null. emit_struct_zero
                            // descends through nested struct fields.
                            int32_t *zprefix = arena_new_array(e->arena, int32_t, 1);
                            zprefix[0] = 0;
                            emit_struct_zero(e, pelem, st_ty, sd, zprefix, 1);
                            if (elem->kind == EX_INT && elem->int_value == 0) {
                                continue;
                            }
                            if (elem->kind == EX_COMPOUND) {
                                // Element is a nested brace list — fill in
                                // each field positionally / by designator.
                                size_t fn = elem->args.size;
                                if (fn > sd->fields.size) {
                                    EMIT_ERR(e, "too many initializers for "
                                                "array element {}", (int64_t)k);
                                    fn = sd->fields.size;
                                }
                                for (size_t i = 0; i < fn; i++) {
                                    int fidx = (int)i;
                                    if (elem->compound_field_names) {
                                        string fname = elem->compound_field_names[i];
                                        if (fname.size != 0) {
                                            fidx = struct_field_index(sd, fname);
                                            if (fidx < 0) {
                                                EMIT_ERR(e, "unknown struct field "
                                                            "{} in initializer", fname);
                                                continue;
                                            }
                                        }
                                    }
                                    Type ft = sd->fields.data[fidx].type;
                                    int32_t fp[2] = {0, (int32_t)fidx};
                                    MLIR_ValueHandle pf = emit_gep(
                                        e, pelem, st_ty, fp, 2, NULL, 0);
                                    Expr *fe = elem->args.data[i];
                                    if (ft.kind == TY_STRUCT) {
                                        StructDef *fsd = find_struct(e, ft.struct_name);
                                        if (!fsd) {
                                            EMIT_ERR(e, "unknown struct type "
                                                        "for nested-struct field "
                                                        "initializer");
                                            continue;
                                        }
                                        if (fe->kind == EX_INT && fe->int_value == 0) {
                                            continue;
                                        }
                                        if (fe->kind == EX_COMPOUND &&
                                            fe->cast_type.kind == TY_VOID) {
                                            fe->cast_type = ft;
                                        }
                                        StructDef *src_sd = NULL;
                                        MLIR_ValueHandle src = resolve_struct_source(
                                            e, sc, fe, &src_sd);
                                        if (src != MLIR_INVALID_HANDLE &&
                                            src_sd == fsd) {
                                            emit_struct_copy(e, pf, src, fsd);
                                        } else {
                                            EVal vv = emit_expr(e, sc, fe);
                                            if (vv.is_ptr && vv.val != MLIR_INVALID_HANDLE) {
                                                emit_struct_copy(e, pf, vv.val, fsd);
                                            } else {
                                                EMIT_ERR(e, "nested-struct field "
                                                            "initializer needs a "
                                                            "struct value");
                                            }
                                        }
                                        continue;
                                    }
                                    EVal v = emit_expr(e, sc, fe);
                                    MLIR_ValueHandle sv;
                                    if (ft.kind == TY_PTR_I32 ||
                                        ft.kind == TY_PTR_CHAR ||
                                        ft.kind == TY_PTR_VOID ||
                                        ft.kind == TY_PTR_STRUCT ||
                                        ft.kind == TY_FNPTR ||
                                        ft.kind == TY_PTR_PTR) {
                                        sv = v.is_ptr ? v.val : emit_null_ptr(e);
                                    } else if (ft.kind == TY_I64) {
                                        sv = coerce_eval(e, v, e->i64);
                                    } else if (ft.kind == TY_F64) {
                                        sv = coerce_eval(e, v, e->f64);
                                    } else {
                                        sv = coerce_eval(
                                            e, v, scalar_mlir_type(e, ft.kind));
                                    }
                                    emit_store_v(e, sv, pf);
                                }
                            } else {
                                // Element is a single struct value (e.g.
                                // copy from another struct lvalue): use
                                // resolve_struct_source + emit_struct_copy.
                                StructDef *src_sd = NULL;
                                MLIR_ValueHandle src = resolve_struct_source(
                                    e, sc, elem, &src_sd);
                                if (src != MLIR_INVALID_HANDLE && src_sd == sd) {
                                    emit_struct_copy(e, pelem, src, sd);
                                } else {
                                    EMIT_ERR(e, "array-of-struct initializer "
                                                "element must be a brace list "
                                                "or struct value");
                                }
                            }
                        }
                    }
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
                    // Struct rhs from a struct-returning call: `S x = f(...)`.
                    if (st->decl_init->kind == EX_CALL) {
                        FuncSig *sig = find_sig(e, st->decl_init->callee);
                        if (sig && sig->ret.type.kind == TY_STRUCT &&
                            sig->ret.sdef == sd) {
                            emit_flat_call(e, sc, sig, st->decl_init->args, NULL, sy->addr);
                            return;
                        }
                    }
                    // Struct rhs from another struct lvalue or *p:
                    // `S x = other;` / `S x = ht->buckets[i].field;` /
                    // `S x = *p;` — copy the struct field-by-field.
                    if (st->decl_init->kind != EX_COMPOUND) {
                        StructDef *src_sd = NULL;
                        MLIR_ValueHandle src = resolve_struct_source(
                            e, sc, st->decl_init, &src_sd);
                        if (src != MLIR_INVALID_HANDLE && src_sd == sd) {
                            emit_struct_copy(e, sy->addr, src, sd);
                            return;
                        }
                    }
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
                            if (ft.kind == TY_STRUCT) {
                                // Nested struct field: copy from the
                                // initializer's source struct field-by-field
                                // (a single emit_store_v would only copy the
                                // first 8 bytes).
                                StructDef *fsd = find_struct(e, ft.struct_name);
                                if (!fsd) {
                                    EMIT_ERR(e, "unknown struct type for "
                                                "nested-struct field "
                                                "initializer");
                                    continue;
                                }
                                Expr *ae = st->decl_init->args.data[i];
                                if (ae->kind == EX_INT && ae->int_value == 0) {
                                    continue; // already zero-initialised
                                }
                                // Nested aggregate `{ ... }` for this
                                // struct field: parser leaves cast_type
                                // unset (TY_VOID); fill it in here so
                                // emit_expr's EX_COMPOUND path resolves
                                // the field's struct type.
                                if (ae->kind == EX_COMPOUND &&
                                    ae->cast_type.kind == TY_VOID) {
                                    ae->cast_type = ft;
                                }
                                StructDef *src_sd = NULL;
                                MLIR_ValueHandle src = resolve_struct_source(
                                    e, sc, ae, &src_sd);
                                if (src != MLIR_INVALID_HANDLE && src_sd == fsd) {
                                    emit_struct_copy(e, p, src, fsd);
                                } else {
                                    EVal vv = emit_expr(e, sc, ae);
                                    if (vv.is_ptr && vv.val != MLIR_INVALID_HANDLE) {
                                        emit_struct_copy(e, p, vv.val, fsd);
                                    } else {
                                        EMIT_ERR(e, "nested-struct field "
                                                    "initializer needs a "
                                                    "struct value");
                                    }
                                }
                                continue;
                            }
                            EVal v = emit_expr(
                                e, sc, st->decl_init->args.data[i]);
                            MLIR_ValueHandle sv;
                            if (ft.kind == TY_PTR_I32 ||
                                ft.kind == TY_PTR_CHAR ||
                                ft.kind == TY_PTR_VOID ||
                                ft.kind == TY_PTR_STRUCT ||
                                ft.kind == TY_FNPTR ||
                                ft.kind == TY_PTR_PTR) {
                                if (v.is_ptr) {
                                    sv = v.val;
                                } else {
                                    // Literal 0 (or any non-ptr) for a
                                    // pointer field: emit a real null ptr
                                    // so the store is 8 bytes wide.
                                    sv = emit_null_ptr(e);
                                }
                            } else if (ft.kind == TY_I64) {
                                sv = coerce_eval(e, v, e->i64);
                            } else if (ft.kind == TY_F64) {
                                sv = coerce_eval(e, v, e->f64);
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
                    MLIR_ValueHandle iv = coerce_eval(e, v, e->f32);
                    emit_store_v(e, iv, sy->addr);
                } else {
                    emit_store_v(e, emit_const_f32(e, 0.0), sy->addr);
                }
            } else if (st->decl_type.kind == TY_F64) {
                sy->addr = emit_alloca(e, e->f64);
                if (st->decl_init) {
                    EVal v = emit_expr(e, sc, st->decl_init);
                    MLIR_ValueHandle iv = coerce_eval(e, v, e->f64);
                    emit_store_v(e, iv, sy->addr);
                } else {
                    emit_store_v(e, emit_const_f64(e, 0.0), sy->addr);
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
                } else if (st->expr->kind == EX_COMPOUND &&
                           st->expr->cast_type.kind == TY_STRUCT) {
                    EVal cv = emit_expr(e, sc, st->expr);
                    if (cv.sdef != want) {
                        EMIT_ERR(e, "returned compound literal type mismatch");
                    } else {
                        emit_struct_copy(e, out, cv.val, want);
                    }
                } else {
                    StructDef *src_sd = NULL;
                    MLIR_ValueHandle src = resolve_struct_source(e, sc, st->expr, &src_sd);
                    if (src != MLIR_INVALID_HANDLE && src_sd == want) {
                        emit_struct_copy(e, out, src, want);
                    } else {
                        EMIT_ERR(e, "struct return must be a variable or struct call");
                    }
                }
                emit_return_op(e, NULL, 0);
                e->terminated = true;
                return;
            }
            MLIR_TypeHandle want_ty;
            if (sig && sig->ret.type.kind == TY_F32) want_ty = e->f32;
            else if (sig && sig->ret.type.kind == TY_F64) want_ty = e->f64;
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
        case ST_LABEL: {
            struct LabelBlock *lb = e->labels;
            while (lb && !str_eq(lb->name, st->label_name)) lb = lb->next;
            if (!lb) {
                EMIT_ERR(e, "internal: label not pre-registered");
                return;
            }
            // Block was already appended to the func region in
            // collect_labels; just terminate the predecessor and
            // continue emission inside the label block.
            if (!e->terminated) emit_branch(e, lb->block);
            e->cur_block = lb->block;
            e->terminated = false;
            return;
        }
        case ST_GOTO: {
            struct LabelBlock *lb = e->labels;
            while (lb && !str_eq(lb->name, st->label_name)) lb = lb->next;
            if (!lb) {
                EMIT_ERR(e, "goto: undefined label '{}'", st->label_name);
                return;
            }
            emit_branch(e, lb->block);
            return;
        }
        case ST_PRINT: {
            EVal v = emit_expr(e, sc, st->expr);
            if (e->target_wasm32) {
                if (v.is_str) {
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
                if (v.is_float && v.is_f64) {
                    ops[0] = emit_fptrunc_f64_to_f32(e, v.val);
                } else {
                    ops[0] = v.val;
                }
                emit_op(e, OP_TYPE_VECTOR_PRINT, str_lit("vector.print"),
                        NULL, 0, NULL, 0, ops, 1, NULL, 0, NULL, 0);
                return;
            }
            if (v.is_str) {
                MLIR_ValueHandle fmt = emit_string_ptr(e, str_lit("%s\n\0"));
                emit_printf_call(e, fmt, v.val, true);
                return;
            }
            MLIR_ValueHandle arg = v.val;
            MLIR_ValueHandle fmt;
            MLIR_TypeHandle arg_ty;
            if (v.is_float && v.is_f64) {
                fmt = emit_string_ptr(e, str_lit("%g\n\0"));
                arg_ty = e->f64;
            } else if (v.is_float) {
                fmt = emit_string_ptr(e, str_lit("%g\n\0"));
                arg = emit_fpext_f32_to_f64(e, v.val);
                arg_ty = e->f64;
            } else if (v.is_i64) {
                fmt = emit_string_ptr(e, str_lit("%lld\n\0"));
                arg_ty = e->i64;
            } else if (v.is_ptr) {
                fmt = emit_string_ptr(e, str_lit("%lld\n\0"));
                arg = emit_ptrtoint_i64(e, v.val);
                arg_ty = e->i64;
            } else {
                fmt = emit_string_ptr(e, str_lit("%lld\n\0"));
                arg = v.is_unsigned ? emit_extui_i32_to_i64(e, v.val)
                                    : emit_extsi_i32_to_i64(e, v.val);
                arg_ty = e->i64;
            }
            (void)arg_ty;
            emit_printf_call(e, fmt, arg, true);
            return;
        }
        case ST_BLOCK: {
            if (st->block_no_scope) {
                for (size_t i = 0; i < st->block_body.size && !e->terminated; i++)
                    emit_stmt(e, sc, st->block_body.data[i]);
                return;
            }
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
            for (size_t i = 0; i < st->for_steps.size; i++) {
                (void)emit_expr(e, &inner, st->for_steps.data[i]);
            }
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
                MLIR_ValueHandle val = cv.is_i64 ? emit_const_i64(e, c->value)
                                                 : emit_const_i32(e, c->value);
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
    for (size_t i = 0; i < body.size; i++) {
        Stmt *st = body.data[i];
        if (e->terminated && st->kind != ST_LABEL) continue;
        emit_stmt(e, sc, st);
    }
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
               ty.kind != TY_F64 &&
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
        // Always compute llvm_fn_ty too — needed for variadic emit and for
        // static functions which we emit as `llvm.func` with internal linkage.
        {
            MLIR_TypeHandle ret_ty = (sig->n_flat_out == 1)
                ? sig->flat_out_tys[0]
                : MLIR_CreateTypeLLVMVoid(e->ctx);
            sig->llvm_fn_ty = MLIR_CreateTypeLLVMFunction(e->ctx,
                ret_ty, sig->flat_in_tys, sig->n_flat_in,
                /*is_var_arg=*/sig->is_variadic);
        }
    }
}

// Recursively walk all statements in a function body and pre-allocate
// a block for every ST_LABEL encountered. The blocks are appended to
// e->labels in source order; emit-time the corresponding ST_LABEL site
// appends each block to the function region.
static void collect_labels(E *e, VecStmtPtr body) {
    for (size_t i = 0; i < body.size; i++) {
        Stmt *st = body.data[i];
        if (!st) continue;
        switch (st->kind) {
            case ST_LABEL: {
                struct LabelBlock *lb = arena_new(e->arena, struct LabelBlock);
                lb->name = st->label_name;
                lb->block = MLIR_CreateBlock(e->ctx);
                MLIR_AppendRegionBlock(e->ctx, e->func_region, lb->block);
                lb->next = e->labels;
                e->labels = lb;
                break;
            }
            case ST_IF:
                collect_labels(e, st->then_body);
                if (st->has_else) collect_labels(e, st->else_body);
                break;
            case ST_WHILE:
            case ST_DO_WHILE:
                collect_labels(e, st->while_body);
                break;
            case ST_FOR:
                collect_labels(e, st->for_body);
                break;
            case ST_BLOCK:
            case ST_SWITCH:
                collect_labels(e, st->block_body);
                break;
            default:
                break;
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
    e->entry_block = entry;
    e->entry_const_one = emit_const_i64(e, 1);
    // entry_const_one is op #0 in entry; allocas hoisted from inner blocks
    // get inserted starting at index 1 to ensure they dominate all uses.
    e->entry_alloca_insert_idx = 1;
    e->func_region = body_r;
    e->cur_sig = sig;
    e->cur_sret_ptr = MLIR_INVALID_HANDLE;
    e->terminated = false;
    e->next_ssa = 0;
    e->labels = NULL;
    // Pre-pass: walk the function body and pre-allocate a block for each
    // ST_LABEL so that forward `goto` references resolve cleanly.
    collect_labels(e, f->body);

    size_t arg_off = 0;
    if (sig->sret) {
        e->cur_sret_ptr = flat_args[arg_off++];
    }

    for (size_t i = 0; i < sig->n_params; i++) {
        SlotInfo *p = &sig->params[i];
        Sym *sy = arena_new(e->arena, Sym);
        *sy = (Sym){0};
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
        } else if (p->type.kind == TY_VA_LIST) {
            // va_list is passed as the alloca pointer of the caller's
            // 32-byte slot. Reuse it directly so va_arg / va_end operate
            // on the caller's slot.
            sy->addr = blk;
        } else if (p->type.kind == TY_F32) {
            sy->addr = emit_alloca(e, e->f32);
            emit_store_v(e, blk, sy->addr);
        } else if (p->type.kind == TY_F64) {
            sy->addr = emit_alloca(e, e->f64);
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
                : (ty == e->f64) ? emit_const_f64(e, 0.0)
                : (ty == e->i64) ? emit_const_i64(e, 0)
                : (ty == e->ptr) ? emit_null_ptr(e)
                : emit_const_i32(e, 0);
            MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v;
            emit_return_op(e, ops, 1);
        }
        e->terminated = true;
    }
    e->cur_block = saved;
    e->entry_block = MLIR_INVALID_HANDLE;
    e->func_region = saved_region;
    e->cur_sig = saved_sig;
    e->cur_sret_ptr = saved_sret;

    MLIR_AttributeHandle sym_name = MLIR_CreateAttributeString(e->ctx, str_lit("sym_name"), f->name);
    // Variadic functions are emitted as `llvm.func` with an LLVMFunctionType
    // attribute (which carries the var_arg flag); non-variadic ones stay as
    // `func.func` with a regular FunctionType. Static functions add an
    // `llvm.linkage = #llvm.linkage<internal>` attribute that propagates to
    // LLVM IR through convert-func-to-llvm.
    MLIR_TypeHandle ft = sig->is_variadic ? sig->llvm_fn_ty : sig->fn_ty;
    MLIR_AttributeHandle fn_ty_attr = MLIR_CreateAttributeType(e->ctx, str_lit("function_type"), ft);
    // Up to 4 attrs: sym_name, function_type, llvm.linkage, wasm.export_name.
    MLIR_AttributeHandle *attrs = arena_new_array(e->arena, MLIR_AttributeHandle, 4);
    attrs[0] = sym_name; attrs[1] = fn_ty_attr;
    int n_attrs = 2;
    if (f->is_static) {
        attrs[n_attrs++] = MLIR_CreateAttributeLLVMLinkageInternal(e->ctx, str_lit("llvm.linkage"));
    }
    // `__attribute__((__export_name__("foo")))` on a function definition
    // re-exports the function under a different wasm export name (used
    // e.g. by corec's wasm_buddy_alloc / wasm_buddy_free helpers).
    if (f->wasm_export_name.size > 0) {
        attrs[n_attrs++] = MLIR_CreateAttributeString(e->ctx, str_lit("wasm.export_name"),
                                                      f->wasm_export_name);
    }
    MLIR_RegionHandle *regs = arena_new_array(e->arena, MLIR_RegionHandle, 1); regs[0] = body_r;
    MLIR_OpHandle fn;
    if (sig->is_variadic) {
        fn = MLIR_CreateOp(e->ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.func"),
                           attrs, n_attrs, NULL, 0, NULL, 0, NULL, 0,
                           regs, 1, eloc(e, 0), MLIR_INVALID_HANDLE,
                           str_lit(""), -1);
    } else {
        fn = MLIR_CreateOp(e->ctx, OP_TYPE_FUNC_FUNC, str_lit("func.func"),
                           attrs, n_attrs, NULL, 0, NULL, 0, NULL, 0,
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
        case EX_FLOAT: t.kind = ex->is_f64 ? TY_F64 : TY_F32; return t;
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
            if (inner.kind == TY_PTR_I32) {
                if (inner.ptr_is_f32) { t.kind = TY_F32; return t; }
                if (inner.ptr_is_f64) { t.kind = TY_F64; return t; }
                if (inner.ptr_is_i64) { t.kind = TY_I64; return t; }
                t.kind = TY_I32; return t;
            }
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
            if (inner.kind == TY_I64) { t.kind = TY_PTR_I32; t.ptr_is_i64 = true; return t; }
            if (inner.kind == TY_F32) { t.kind = TY_PTR_I32; t.ptr_is_f32 = true; return t; }
            if (inner.kind == TY_F64) { t.kind = TY_PTR_I32; t.ptr_is_f64 = true; return t; }
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
                if (base.kind == TY_PTR_I32 && base.ptr_is_f32) { t.kind = TY_F32; return t; }
                if (base.kind == TY_PTR_I32 && base.ptr_is_f64) { t.kind = TY_F64; return t; }
                t.kind = (base.kind == TY_ARRAY_I32 && base.array_elem_is_i64) ||
                         (base.kind == TY_PTR_I32 && base.ptr_is_i64) ? TY_I64 : TY_I32;
                return t;
            }
            if (base.kind == TY_PTR_CHAR) { t.kind = TY_I32; return t; }
            if (base.kind == TY_ARRAY_PTR_CHAR) { t.kind = TY_PTR_CHAR; return t; }
            if (base.kind == TY_ARRAY_F32) { t.kind = TY_F32; return t; }
            if (base.kind == TY_ARRAY_STRUCT) {
                t.kind = TY_STRUCT; t.struct_name = base.struct_name; return t;
            }
            if (base.kind == TY_PTR_STRUCT) {
                t.kind = TY_STRUCT; t.struct_name = base.struct_name; return t;
            }
            if (base.kind == TY_PTR_PTR && base.pointee) return *base.pointee;
            return t;
        }
        case EX_CALL: {
            FuncSig *sig = find_sig(e, ex->callee);
            if (sig) return sig->ret.type;
            // Indirect call through a fnptr-typed local/param/global.
            Sym *sy = lookup(e, sc, ex->callee);
            if (sy && sy->type.kind == TY_FNPTR && sy->type.fnptr_ret) {
                return *sy->type.fnptr_ret;
            }
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
            if (a.kind == TY_F64 || b.kind == TY_F64) { t.kind = TY_F64; return t; }
            if (a.kind == TY_F32 || b.kind == TY_F32) { t.kind = TY_F32; return t; }
            if (a.kind == TY_I64 || b.kind == TY_I64) { t.kind = TY_I64; return t; }
            return a;
        }
        case EX_UN:
        case EX_ASSIGN: {
            // Use the LHS type as the result type.
            return infer_expr_type(e, sc, ex->lhs ? ex->lhs : ex->lvalue);
        }
        case EX_TERNARY: {
            // For `c ? a : b` the result type is the type of the
            // then/else branches (`a` / `b`), NOT the condition `c`.
            // tinyc's AST stores: lhs=cond, rhs=then, lvalue=else.
            // Pick the then-branch (rhs) first; fall back to lvalue or
            // lhs only if rhs is missing (defensive).
            Expr *val = ex->rhs ? ex->rhs : (ex->lvalue ? ex->lvalue : ex->lhs);
            return infer_expr_type(e, sc, val);
        }
        default: return t;
    }
}

static int64_t type_size(E *e, Type t) {
    int64_t ptr_sz = e->target_wasm32 ? 4 : 8;
    if (t.kind == TY_I32) return 4;
    if (t.kind == TY_I64) return 8;
    if (t.kind == TY_F32) return 4;
    if (t.kind == TY_F64) return 8;
    if (t.kind == TY_PTR_I32 || t.kind == TY_PTR_STRUCT) return ptr_sz;
    if (t.kind == TY_PTR_CHAR || t.kind == TY_PTR_VOID || t.kind == TY_FNPTR) return ptr_sz;
    if (t.kind == TY_PTR_PTR) return ptr_sz;
    if (t.kind == TY_ARRAY_I32) {
        int64_t es = t.array_elem_is_i64 ? 8 : t.array_elem_is_i8 ? 1 : 4;
        return es * t.array_len * (t.array_len2 ? t.array_len2 : 1);
    }
    if (t.kind == TY_ARRAY_F32) return 4 * t.array_len * (t.array_len2 ? t.array_len2 : 1);
    if (t.kind == TY_ARRAY_PTR_STRUCT || t.kind == TY_ARRAY_PTR_CHAR)
        return ptr_sz * t.array_len;
    if (t.kind == TY_STRUCT) {
        StructDef *sd = find_struct(e, t.struct_name);
        if (!sd) return 0;
        if (sd->is_union) {
            MLIR_TypeHandle elem; int64_t es, c;
            union_blob_layout(e, sd, &elem, &es, &c);
            return es * c;
        }
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
        // Inline the TY_STRUCT branch by name rather than recursing
        // with a Type-by-value parameter — passing a Type with
        // `kind == TY_STRUCT` re-into `type_size` (which itself takes
        // `Type` by value) hit a clang -Os miscompile on wasm32-wasi
        // that returned 0 for the array's element size.
        StructDef *sd = find_struct(e, t.struct_name);
        if (!sd) return 0;
        if (sd->is_union) {
            MLIR_TypeHandle elem; int64_t es, c;
            union_blob_layout(e, sd, &elem, &es, &c);
            return es * c * t.array_len;
        }
        int64_t off = 0, max_align = 1;
        for (size_t i = 0; i < sd->fields.size; i++) {
            Type ft = sd->fields.data[i].type;
            int64_t fa = type_align(e, ft);
            int64_t fs = type_size(e, ft);
            if (fa > max_align) max_align = fa;
            off = (off + fa - 1) / fa * fa;
            off += fs;
        }
        if (max_align > 0) off = (off + max_align - 1) / max_align * max_align;
        return off * t.array_len;
    }
    return 0;
}
// C-level `sizeof` of an expression operand. tinyc represents an indexed
// element of a `char`/`unsigned char` array as a 4-byte i32 rvalue, so
// infer_expr_type + type_size would wrongly report 4; C mandates 1 for a
// char element. Detect that case directly off the array type and report 1
// without perturbing infer_expr_type's rvalue type (which feeds _Generic
// matching and other callers) or type_size (which drives struct layout,
// where scalar char stays a 4-byte i32). All other operands defer to
// type_size, preserving tinyc's prior sizeof behavior exactly.
static int64_t c_sizeof_expr(E *e, Scope *sc, Expr *ex) {
    if (ex->kind == EX_INDEX) {
        Type base = infer_expr_type(e, sc, ex->lhs);
        if (base.kind == TY_ARRAY_I32 && base.array_elem_is_i8) return 1;
    }
    return type_size(e, infer_expr_type(e, sc, ex));
}
static int64_t type_align(E *e, Type t) {
    int64_t ptr_a = e->target_wasm32 ? 4 : 8;
    if (t.kind == TY_I32 || t.kind == TY_F32) return 4;
    if (t.kind == TY_I64 || t.kind == TY_F64) return 8;
    if (t.kind == TY_PTR_I32 || t.kind == TY_PTR_STRUCT) return ptr_a;
    if (t.kind == TY_PTR_CHAR || t.kind == TY_PTR_VOID || t.kind == TY_FNPTR) return ptr_a;
    if (t.kind == TY_PTR_PTR) return ptr_a;
    if (t.kind == TY_ARRAY_I32 || t.kind == TY_ARRAY_F32) return 4;
    if (t.kind == TY_ARRAY_PTR_STRUCT || t.kind == TY_ARRAY_PTR_CHAR) return ptr_a;
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

static void union_blob_layout(E *e, StructDef *sd, MLIR_TypeHandle *elem,
                              int64_t *elem_size, int64_t *count) {
    int64_t max_sz = 0, max_al = 1;
    for (size_t i = 0; i < sd->fields.size; i++) {
        Type ft = sd->fields.data[i].type;
        int64_t s = type_size(e, ft);
        int64_t a = type_align(e, ft);
        if (s > max_sz) max_sz = s;
        if (a > max_al) max_al = a;
    }
    MLIR_TypeHandle et;
    int64_t es;
    if (max_al >= 8) { et = e->i64; es = 8; }
    else if (max_al >= 4) { et = e->i32; es = 4; }
    else { et = e->i8; es = 1; }
    int64_t c = (max_sz + es - 1) / es;
    if (c < 1) c = 1;
    *elem = et;
    *elem_size = es;
    *count = c;
}


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
        if (sd->is_union) {
            // A union overlaps all members at offset 0. Emit its body as a
            // single blob field (array of alignment-wide integers) sized to
            // the largest member, so the llvm.struct size equals max(member
            // sizes) rather than their sum. All union member accesses are
            // normalised to offset 0 elsewhere (see walk_struct_lhs).
            MLIR_TypeHandle uelem; int64_t ues, uc;
            union_blob_layout(e, sd, &uelem, &ues, &uc);
            MLIR_TypeHandle *ubody = arena_new_array(e->arena, MLIR_TypeHandle, 1);
            ubody[0] = MLIR_CreateTypeLLVMArray(e->ctx, uelem, (uint64_t)uc);
            MLIR_SetTypeLLVMStructBody(e->ctx, e->struct_types[i].ty, ubody, 1);
            continue;
        }
        size_t n = sd->fields.size;
        MLIR_TypeHandle *body = arena_new_array(e->arena, MLIR_TypeHandle, n ? n : 1);
        for (size_t k = 0; k < n; k++) {
            Type ft = sd->fields.data[k].type;
            if (ft.kind == TY_I32) body[k] = e->i32;
            else if (ft.kind == TY_I64) body[k] = e->i64;
            else if (ft.kind == TY_F32) body[k] = e->f32;
            else if (ft.kind == TY_F64) body[k] = e->f64;
            else if (ft.kind == TY_PTR_STRUCT || ft.kind == TY_PTR_I32 ||
                     ft.kind == TY_PTR_CHAR || ft.kind == TY_PTR_VOID ||
                     ft.kind == TY_PTR_PTR || ft.kind == TY_FNPTR) body[k] = e->ptr;
            else if (ft.kind == TY_STRUCT) {
                StructDef *inner = find_struct(e, ft.struct_name);
                MLIR_TypeHandle t = find_struct_type(e, inner);
                body[k] = t;
            } else if (ft.kind == TY_ARRAY_I32 || ft.kind == TY_ARRAY_F32) {
                MLIR_TypeHandle elem;
                if (ft.kind == TY_ARRAY_F32) elem = e->f32;
                else if (ft.array_elem_is_i64) elem = e->i64;
                else if (ft.array_elem_is_i8) elem = e->i8;
                else elem = e->i32;
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

static int g_last_emit_errors = 0;

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
    e.f64 = MLIR_CreateTypeFloat(ctx, 64, false);
    e.index = MLIR_CreateTypeIndex(ctx);
    e.ptr = MLIR_CreateTypeLLVMPointer(ctx);
    e.target_wasm32 = program->target_wasm32;
    e.size_t_ty = program->target_wasm32 ? e.i32 : e.i64;
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
        // Record the struct sdef on the symbol so indexed/field accesses
        // through the global (e.g. `g_arr[i].field`) can resolve fields.
        if (g->type.kind == TY_PTR_STRUCT || g->type.kind == TY_ARRAY_STRUCT
                || g->type.kind == TY_STRUCT) {
            sy->sdef = find_struct(&e, g->type.struct_name);
        }
        sy->next = e.globals;
        e.globals = sy;

        if (g->is_extern) {
            // `extern T x;` — emit external linkage with no initializer.
            // Used for libc symbols like `stdout` resolved by the linker.
            MLIR_TypeHandle gty;
            if (g->type.kind == TY_F32) {
                gty = e.f32;
            } else if (g->type.kind == TY_F64) {
                gty = e.f64;
            } else if (g->type.kind == TY_PTR_CHAR || g->type.kind == TY_PTR_I32 ||
                       g->type.kind == TY_PTR_STRUCT || g->type.kind == TY_PTR_VOID ||
                       g->type.kind == TY_FNPTR || g->type.kind == TY_PTR_PTR) {
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
        } else if (g->type.kind == TY_F64) {
            MLIR_OpHandle gop = MLIR_CreateLLVMGlobal(ctx, g->name, e.f64,
                /*is_constant=*/false,
                /*init_kind=*/1, 0, g->has_init ? g->init_float : 0.0,
                NULL, e.loc);
            MLIR_AppendBlockOp(ctx, mb, gop);
        } else if (g->type.kind == TY_PTR_I32 || g->type.kind == TY_PTR_STRUCT ||
                   g->type.kind == TY_PTR_VOID || g->type.kind == TY_FNPTR ||
                   g->type.kind == TY_PTR_PTR) {
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
        } else if (g->type.kind == TY_ARRAY_STRUCT ||
                   g->type.kind == TY_ARRAY_I32 ||
                   g->type.kind == TY_ARRAY_PTR_CHAR ||
                   g->type.kind == TY_ARRAY_PTR_STRUCT) {
            // Zero-initialized aggregate global. Resolve a deferred array
            // length expression (e.g. `[20 + 1]`) at emit time using
            // ast_fold_int with no scope (file-scope const expressions
            // only).
            if (g->type.array_len_expr) {
                int64_t v = 0;
                if (!ast_fold_int(&e, NULL, g->type.array_len_expr, &v)) {
                    EMIT_ERR(&e, "non-constant global array length");
                }
                g->type.array_len = v;
            }
            MLIR_TypeHandle elem;
            if (g->type.kind == TY_ARRAY_STRUCT) {
                StructDef *sd = find_struct(&e, g->type.struct_name);
                if (!sd) { EMIT_ERR(&e, "unknown struct in global array"); continue; }
                elem = find_struct_type(&e, sd);
            } else if (g->type.kind == TY_ARRAY_PTR_CHAR ||
                       g->type.kind == TY_ARRAY_PTR_STRUCT) {
                elem = e.ptr;
            } else {
                elem = g->type.array_elem_is_i64 ? e.i64
                     : g->type.array_elem_is_i8  ? e.i8
                     : e.i32;
            }
            MLIR_TypeHandle arr_ty = MLIR_CreateTypeLLVMArray(
                ctx, elem, (uint64_t)g->type.array_len);
            MLIR_OpHandle gop;
            if (g->init_array_data.size > 0 &&
                g->type.kind == TY_ARRAY_I32) {
                // Non-zero aggregate initializer: emit the raw bytes
                // as the global's `value` STRING attribute. lower_global
                // (wasm) and emit_global (native) both interpret it as
                // packed little-endian element bytes.
                gop = MLIR_CreateLLVMGlobalArrayInit(ctx, g->name, arr_ty,
                    /*is_constant=*/false,
                    g->init_array_data, e.loc);
            } else {
                MLIR_BlockHandle init_blk = MLIR_INVALID_HANDLE;
                gop = MLIR_CreateLLVMGlobal(ctx, g->name, arr_ty,
                    /*is_constant=*/false,
                    /*init_kind=*/2, 0, 0.0, &init_blk, e.loc);
                MLIR_BlockHandle save_blk = e.cur_block;
                bool save_term = e.terminated;
                e.cur_block = init_blk;
                e.terminated = false;
                // Materialize a zero of the array type.
                MLIR_ValueHandle zv = MLIR_CreateValueOpResult(e.ctx, MLIR_INVALID_HANDLE, 0,
                                                              arr_ty, ssa_name(&e), eloc(&e, 0));
                MLIR_TypeHandle *zrts = arena_new_array(e.arena, MLIR_TypeHandle, 1); zrts[0] = arr_ty;
                MLIR_ValueHandle *zrs = arena_new_array(e.arena, MLIR_ValueHandle, 1); zrs[0] = zv;
                emit_op(&e, OP_TYPE_LLVM_MLIR_ZERO, str_lit("llvm.mlir.zero"),
                        zrts, 1, zrs, 1, NULL, 0, NULL, 0, NULL, 0);
                MLIR_ValueHandle *rops = arena_new_array(arena, MLIR_ValueHandle, 1);
                rops[0] = zv;
                emit_op(&e, OP_TYPE_LLVM_RETURN, str_lit("llvm.return"),
                        NULL, 0, NULL, 0, rops, 1, NULL, 0, NULL, 0);
                e.cur_block = save_blk;
                e.terminated = save_term;
            }
            // Record the struct sdef on the symbol for indexed access.
            if (g->type.kind == TY_ARRAY_STRUCT) {
                sy->sdef = find_struct(&e, g->type.struct_name);
            }
            MLIR_AppendBlockOp(ctx, mb, gop);
        } else if (g->type.kind == TY_I64) {
            MLIR_OpHandle gop = MLIR_CreateLLVMGlobal(ctx, g->name, e.i64,
                /*is_constant=*/false,
                /*init_kind=*/0, g->has_init ? g->init_int : 0, 0.0,
                NULL, e.loc);
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
        // Skip extern decls for forward-declared functions nobody
        // actually called. Without this, every prototype reachable
        // through an `#include` becomes a wasm import that the link
        // step can't resolve.
        if (!sig->is_used) continue;
        MLIR_AttributeHandle a0 = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), fwd->name);
        MLIR_TypeHandle ft = sig->is_variadic ? sig->llvm_fn_ty : sig->fn_ty;
        MLIR_AttributeHandle a1 = MLIR_CreateAttributeType(ctx, str_lit("function_type"), ft);
        MLIR_AttributeHandle a2 = MLIR_CreateAttributeString(ctx, str_lit("sym_visibility"), str_lit("private"));
        // Forward the `__attribute__((__import_module__("...")))` /
        // `__import_name__("...")` annotations onto the MLIR func op
        // so the wasm binary emitter can place this import into the
        // requested module (e.g. WASI's `wasi_snapshot_preview1`).
        size_t n_xtra = 0;
        if (fwd->wasm_import_module.size > 0) n_xtra++;
        if (fwd->wasm_import_name.size > 0) n_xtra++;
        MLIR_AttributeHandle *attrs = arena_new_array(arena, MLIR_AttributeHandle, 3 + n_xtra);
        attrs[0] = a0; attrs[1] = a1; attrs[2] = a2;
        size_t na = 3;
        if (fwd->wasm_import_module.size > 0) {
            attrs[na++] = MLIR_CreateAttributeString(ctx, str_lit("wasm.import_module"),
                                                    fwd->wasm_import_module);
        }
        if (fwd->wasm_import_name.size > 0) {
            attrs[na++] = MLIR_CreateAttributeString(ctx, str_lit("wasm.import_name"),
                                                    fwd->wasm_import_name);
        }
        MLIR_RegionHandle body = MLIR_CreateRegion(ctx);
        MLIR_RegionHandle *regs = arena_new_array(arena, MLIR_RegionHandle, 1); regs[0] = body;
        MLIR_OpHandle decl;
        if (sig->is_variadic) {
            decl = MLIR_CreateOp(ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.func"),
                                 attrs, na, NULL, 0, NULL, 0, NULL, 0,
                                 regs, 1, e.loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
        } else {
            decl = MLIR_CreateOp(ctx, OP_TYPE_FUNC_FUNC, str_lit("func.func"),
                                 attrs, na, NULL, 0, NULL, 0, NULL, 0,
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
        // Skip the auto-decl if the user already has a forward decl OR a
        // full definition (corec-stdlib's stdlib/stdlib.c defines them).
        if (str_eq(fwd->name, str_lit("malloc"))) have_user_malloc = true;
        else if (str_eq(fwd->name, str_lit("free"))) have_user_free = true;
    }
    if (!have_user_malloc) {
        MLIR_TypeHandle *ins = arena_new_array(arena, MLIR_TypeHandle, 1); ins[0] = e.size_t_ty;
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

    // __tinyc_syscall6 — the raw-syscall intrinsic stub. Declared (i64 x7) ->
    // i64; the native llvm->aarch64 backend synthesises its `svc` body. Only
    // emitted when user code used __builtin_syscall6.
    if (e.need_syscall6_stub) {
        MLIR_TypeHandle *ins = arena_new_array(arena, MLIR_TypeHandle, 7);
        for (size_t k = 0; k < 7; k++) ins[k] = e.i64;
        MLIR_TypeHandle *outs = arena_new_array(arena, MLIR_TypeHandle, 1); outs[0] = e.i64;
        MLIR_TypeHandle fty = MLIR_CreateTypeFunction(ctx, ins, 7, outs, 1);
        MLIR_AttributeHandle a0 = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), str_lit("__tinyc_syscall6"));
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

    // tinyc_va_arg_* helper declarations — emitted on demand when user code
    // uses va_arg, unless the generated single-TU root already defines them.
    if (e.need_va_arg_helpers) {
        struct { string name; MLIR_TypeHandle ret; } helpers[] = {
            { str_lit("tinyc_va_arg_i32"), e.i32 },
            { str_lit("tinyc_va_arg_i64"), e.i64 },
            { str_lit("tinyc_va_arg_f64"), e.f64 },
            { str_lit("tinyc_va_arg_ptr"), e.ptr },
        };
        for (size_t k = 0; k < 4; k++) {
            bool have_helper = false;
            for (size_t i = 0; i < program->funcs.size; i++) {
                Func *f = program->funcs.data[i];
                if (!f->is_forward && str_eq(f->name, helpers[k].name)) {
                    have_helper = true;
                    break;
                }
            }
            if (have_helper) continue;
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

    if (e.need_va_arg_struct) {
        bool have_va_arg_struct = false;
        for (size_t i = 0; i < program->funcs.size; i++) {
            Func *f = program->funcs.data[i];
            if (!f->is_forward && str_eq(f->name, str_lit("tinyc_va_arg_struct"))) {
                have_va_arg_struct = true;
                break;
            }
        }
        if (!have_va_arg_struct) {
        // void tinyc_va_arg_struct(va_list *ap, void *out, i64 size)
        MLIR_TypeHandle *ins = arena_new_array(arena, MLIR_TypeHandle, 3);
        ins[0] = e.ptr; ins[1] = e.ptr; ins[2] = e.i64;
        MLIR_TypeHandle fty = MLIR_CreateTypeFunction(ctx, ins, 3, NULL, 0);
        MLIR_AttributeHandle a0 = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), str_lit("tinyc_va_arg_struct"));
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

    if (e.need_printf_decl) {
        bool have_printf = false;
        for (size_t i = 0; i < program->funcs.size; i++) {
            Func *f = program->funcs.data[i];
            if (str_eq(f->name, str_lit("printf"))) {
                have_printf = true;
                break;
            }
        }
        if (!have_printf) {
            MLIR_TypeHandle *ins = arena_new_array(arena, MLIR_TypeHandle, 1);
            ins[0] = e.ptr;
            MLIR_TypeHandle fty = MLIR_CreateTypeLLVMFunction(ctx, e.i32, ins, 1, true);
            MLIR_AttributeHandle a0 = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), str_lit("printf"));
            MLIR_AttributeHandle a1 = MLIR_CreateAttributeType(ctx, str_lit("function_type"), fty);
            MLIR_AttributeHandle *attrs = arena_new_array(arena, MLIR_AttributeHandle, 2);
            attrs[0] = a0; attrs[1] = a1;
            MLIR_RegionHandle body = MLIR_CreateRegion(ctx);
            MLIR_RegionHandle *regs = arena_new_array(arena, MLIR_RegionHandle, 1); regs[0] = body;
            MLIR_OpHandle decl = MLIR_CreateOp(ctx, OP_TYPE_UNREGISTERED, str_lit("llvm.func"),
                                               attrs, 2, NULL, 0, NULL, 0, NULL, 0,
                                               regs, 1, e.loc, MLIR_INVALID_HANDLE, str_lit(""), -1);
            MLIR_AppendBlockOp(ctx, mb, decl);
        }
    }

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

    // Emit any string-literal globals collected by ST_PRINT/EX_STR.
    for (size_t i = 0; i < e.n_strings; i++) {
        StringEntry *se = &e.strings[i];
        MLIR_OpHandle gop = MLIR_CreateLLVMGlobalString(
            ctx, se->sym, se->bytes, e.loc);
        MLIR_AppendBlockOp(ctx, mb, gop);
    }

    g_last_emit_errors = e.err_count;
    return module;
}

int tinyc_last_emit_errors(void) { return g_last_emit_errors; }

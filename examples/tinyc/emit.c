// tinyC -> MLIR emitter. Uses ONLY mlir_api.h (plus corec primitives).
// No upstream MLIR headers are included from this directory.
//
// Control flow is emitted in cf-form: every `if`, `while`, `for`,
// `break`, `continue`, and early `return` lowers to manually-created
// basic blocks linked together with cf.br / cf.cond_br terminators.
// Short-circuit && / || keep using scf.if-with-result inside the
// expression itself (no early-exit semantics, simpler than a CFG-level
// rewrite). The lowering pipeline runs SCFToCF before CFToLLVM.

#include "tinyc.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <base/arena.h>
#include <base/format.h>
#include <base/io.h>
#include <base/string.h>

// ----- symbol table -----

typedef struct Sym {
    string name;
    Type   type;                // declared type
    MLIR_ValueHandle addr;      // memref handle backing this local
    // For TY_STRUCT: per-field memref handles (parallel to sdef->fields).
    // For non-struct types: NULL / unused.
    StructDef *sdef;
    MLIR_ValueHandle *field_addrs;
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
    MLIR_BlockHandle continue_block;   // where `continue` jumps
    MLIR_BlockHandle break_block;      // where `break` jumps
    struct LoopCtx *parent;
} LoopCtx;

typedef struct {
    Type        type;          // declared type of the slot
    StructDef  *sdef;          // resolved if type.kind == TY_STRUCT
    size_t      flat_count;    // 1 for scalar; #fields for struct
    size_t      flat_offset;   // index into flat_in_tys / flat_out_tys
} SlotInfo;

typedef struct FuncSig {
    string         name;
    Func          *func;
    SlotInfo       ret;
    SlotInfo      *params;
    size_t         n_params;
    MLIR_TypeHandle *flat_in_tys;
    size_t         n_flat_in;
    MLIR_TypeHandle *flat_out_tys;
    size_t         n_flat_out;
    MLIR_TypeHandle fn_ty;
} FuncSig;

typedef struct {
    MLIR_Context *ctx;
    Arena *arena;
    MLIR_TypeHandle i32;
    MLIR_TypeHandle i1;
    MLIR_TypeHandle f32;
    MLIR_TypeHandle index;
    MLIR_TypeHandle memref_i32;       // memref<i32>  (rank-0)
    MLIR_TypeHandle memref_f32;       // memref<f32>  (rank-0)
    MLIR_BlockHandle cur_block;       // where the next op gets appended
    bool terminated;                  // current block already has a terminator
    MLIR_RegionHandle func_region;    // body region of the enclosing func.func
    MLIR_LocationHandle loc;
    int next_ssa;                     // running %N counter
    LoopCtx *loops;
    Program *program;                 // for resolving struct names
    FuncSig *sigs;                    // signatures of all program functions
    size_t   n_sigs;
    FuncSig *cur_sig;                 // signature of the function being emitted
} E;

static StructDef *find_struct(E *e, string name) {
    if (!e->program) return NULL;
    for (size_t i = 0; i < e->program->structs.size; i++) {
        StructDef *sd = e->program->structs.data[i];
        if (str_eq(sd->name, name)) return sd;
    }
    return NULL;
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

static MLIR_TypeHandle scalar_mlir_type(E *e, TypeKind k) {
    return (k == TY_F32) ? e->f32 : e->i32;
}

static MLIR_LocationHandle eloc(E *e, int line) {
    (void)line;
    return e->loc;
}

static string ssa_name(E *e) {
    int n = e->next_ssa++;
    return format(e->arena, str_lit("%{}"), (int64_t)n);
}

// Append a CFG block to the enclosing func.func body region.
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

// ----- constants and arithmetic helpers -----

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

static MLIR_ValueHandle emit_const_index(E *e, int64_t v) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->index, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->index;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_AttributeHandle val = MLIR_CreateAttributeInteger(e->ctx, str_lit("value"), v, e->index);
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1); as[0] = val;
    emit_op(e, OP_TYPE_INDEX_CONSTANT, str_lit("arith.constant"),
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

static MLIR_ValueHandle emit_alloc(E *e, MLIR_TypeHandle memref_ty) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  memref_ty, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = memref_ty;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    emit_op(e, OP_TYPE_MEMREF_ALLOC, str_lit("memref.alloc"),
            rts, 1, rs, 1, NULL, 0, NULL, 0, NULL, 0);
    return r;
}

static void emit_store(E *e, MLIR_ValueHandle val, MLIR_ValueHandle addr,
                       MLIR_ValueHandle *idxs, size_t n_idxs) {
    size_t n = 2 + n_idxs;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, n);
    ops[0] = val; ops[1] = addr;
    for (size_t i = 0; i < n_idxs; i++) ops[2 + i] = idxs[i];
    emit_op(e, OP_TYPE_MEMREF_STORE, str_lit("memref.store"),
            NULL, 0, NULL, 0, ops, n, NULL, 0, NULL, 0);
}

static MLIR_ValueHandle emit_load(E *e, MLIR_ValueHandle addr, MLIR_TypeHandle elem_ty,
                                  MLIR_ValueHandle *idxs, size_t n_idxs) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  elem_ty, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = elem_ty;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    size_t n = 1 + n_idxs;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, n);
    ops[0] = addr;
    for (size_t i = 0; i < n_idxs; i++) ops[1 + i] = idxs[i];
    emit_op(e, OP_TYPE_MEMREF_LOAD, str_lit("memref.load"),
            rts, 1, rs, 1, ops, n, NULL, 0, NULL, 0);
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

// arith.index_cast i32 -> index
static MLIR_ValueHandle emit_index_cast(E *e, MLIR_ValueHandle i32_val) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->index, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->index;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = i32_val;
    emit_op(e, OP_TYPE_ARITH_INDEX_CAST, str_lit("arith.index_cast"),
            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
    return r;
}

// arith.sitofp i32 -> f32
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

// MLIR cmpi predicate codes (arith dialect):
//   0=eq, 1=ne, 2=slt, 3=sle, 4=sgt, 5=sge
static int64_t cmpi_pred_for(BinOp op) {
    switch (op) {
        case OP_EQ: return 0;
        case OP_NE: return 1;
        case OP_LT: return 2;
        case OP_LE: return 3;
        case OP_GT: return 4;
        case OP_GE: return 5;
        default:    return 0;
    }
}

// MLIR cmpf predicate codes (ordered, quiet): 1=oeq, 6=one, 4=olt, 5=ole, 2=ogt, 3=oge
static int64_t cmpf_pred_for(BinOp op) {
    switch (op) {
        case OP_EQ: return 1;
        case OP_NE: return 6;
        case OP_LT: return 4;
        case OP_LE: return 5;
        case OP_GT: return 2;
        case OP_GE: return 3;
        default:    return 1;
    }
}

// ----- expression / statement emission -----

typedef struct {
    MLIR_ValueHandle val;     // SSA value of the expression's rvalue
    bool is_float;            // true => f32, else i32 (only scalars are produced)
} EVal;

// Forward decl: evaluate an expression as an rvalue.
static EVal emit_expr(E *e, Scope *sc, Expr *ex);

// Evaluate as i32, performing fptosi if necessary.
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

// Coerce an EVal to a target scalar MLIR type (i32 or f32).
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

// Evaluate `ex` as an lvalue: produces (base_addr_memref, optional index value).
// `out_idx` is set to MLIR_INVALID_HANDLE for scalar lvalues; otherwise it's
// an `index`-typed SSA value used as the single index into the rank-1 memref.
typedef struct {
    MLIR_ValueHandle addr;     // memref handle
    MLIR_ValueHandle index;    // index SSA value, or MLIR_INVALID_HANDLE
    MLIR_TypeHandle elem_ty;   // element type (i32 or f32)
} LVal;

static LVal emit_lvalue(E *e, Scope *sc, Expr *ex) {
    LVal r = {.addr = MLIR_INVALID_HANDLE, .index = MLIR_INVALID_HANDLE, .elem_ty = e->i32};
    switch (ex->kind) {
        case EX_VAR: {
            Sym *s = scope_lookup(sc, ex->name);
            if (!s) {
                println(str_lit("tinyc emit: undefined variable {}"), ex->name);
                return r;
            }
            r.addr = s->addr;
            r.elem_ty = (s->type.kind == TY_F32) ? e->f32 : e->i32;
            return r;
        }
        case EX_INDEX: {
            // a[i] : a must be an array variable.
            if (ex->lhs->kind != EX_VAR) {
                println(str_lit("tinyc emit: only simple-array indexing is supported"));
                return r;
            }
            Sym *s = scope_lookup(sc, ex->lhs->name);
            if (!s || s->type.kind != TY_ARRAY_I32) {
                println(str_lit("tinyc emit: indexing of non-array variable"));
                return r;
            }
            MLIR_ValueHandle idx_i32 = emit_expr_i32(e, sc, ex->rhs);
            r.addr = s->addr;
            r.index = emit_index_cast(e, idx_i32);
            r.elem_ty = e->i32;
            return r;
        }
        case EX_DEREF: {
            // *p : evaluate p as an rvalue (a memref<i32> handle).
            if (ex->lhs->kind != EX_VAR) {
                println(str_lit("tinyc emit: only *<var> dereference is supported"));
                return r;
            }
            Sym *s = scope_lookup(sc, ex->lhs->name);
            if (!s || s->type.kind != TY_PTR_I32) {
                println(str_lit("tinyc emit: dereference of non-pointer"));
                return r;
            }
            // The local slot of a pointer variable IS the pointee memref
            // (alias-only pointers — see tinyc.h).
            r.addr = s->addr;
            r.elem_ty = e->i32;
            return r;
        }
        case EX_FIELD: {
            // s.field — base must be a struct variable.
            if (ex->lhs->kind != EX_VAR) {
                println(str_lit("tinyc emit: only <var>.field field access is supported"));
                return r;
            }
            Sym *s = scope_lookup(sc, ex->lhs->name);
            if (!s || s->type.kind != TY_STRUCT || !s->sdef) {
                println(str_lit("tinyc emit: field access on non-struct"));
                return r;
            }
            int idx = struct_field_index(s->sdef, ex->name);
            if (idx < 0) {
                println(str_lit("tinyc emit: unknown struct field {}"), ex->name);
                return r;
            }
            r.addr = s->field_addrs[idx];
            r.elem_ty = (s->sdef->fields.data[idx].kind == TY_F32) ? e->f32 : e->i32;
            return r;
        }
        default:
            println(str_lit("tinyc emit: invalid lvalue"));
            return r;
    }
}

static EVal load_lvalue(E *e, LVal lv) {
    EVal r = {0};
    if (lv.index == MLIR_INVALID_HANDLE) {
        r.val = emit_load(e, lv.addr, lv.elem_ty, NULL, 0);
    } else {
        MLIR_ValueHandle *idxs = arena_new_array(e->arena, MLIR_ValueHandle, 1);
        idxs[0] = lv.index;
        r.val = emit_load(e, lv.addr, lv.elem_ty, idxs, 1);
    }
    r.is_float = (lv.elem_ty == e->f32);
    return r;
}

static void store_lvalue(E *e, LVal lv, MLIR_ValueHandle v) {
    if (lv.index == MLIR_INVALID_HANDLE) {
        emit_store(e, v, lv.addr, NULL, 0);
    } else {
        MLIR_ValueHandle *idxs = arena_new_array(e->arena, MLIR_ValueHandle, 1);
        idxs[0] = lv.index;
        emit_store(e, v, lv.addr, idxs, 1);
    }
}

// Promote a pair to a common arithmetic type (f32 if either side is float).
static void unify_numeric(E *e, EVal *a, EVal *b) {
    if (a->is_float == b->is_float) return;
    if (!a->is_float) { a->val = emit_sitofp(e, a->val); a->is_float = true; }
    if (!b->is_float) { b->val = emit_sitofp(e, b->val); b->is_float = true; }
}

// Load all fields of a struct local into the operand array starting at base.
static void flatten_struct_arg(E *e, Scope *sc, Expr *arg, SlotInfo *p,
                               MLIR_ValueHandle *out, size_t base) {
    if (arg->kind != EX_VAR) {
        println(str_lit("tinyc emit: struct argument must be a struct variable"));
        return;
    }
    Sym *s = scope_lookup(sc, arg->name);
    if (!s || s->type.kind != TY_STRUCT || !s->sdef) {
        println(str_lit("tinyc emit: struct argument {} is not a struct"), arg->name);
        return;
    }
    if (s->sdef != p->sdef) {
        println(str_lit("tinyc emit: struct type mismatch in call argument"));
    }
    for (size_t k = 0; k < s->sdef->fields.size; k++) {
        bool flt = (s->sdef->fields.data[k].kind == TY_F32);
        out[base + k] = emit_load(e, s->field_addrs[k], flt ? e->f32 : e->i32, NULL, 0);
    }
}

// Emit a func.call with flattened arguments and results. If `out_results` is
// non-NULL it is filled with sig->n_flat_out result values.
static void emit_flat_call(E *e, Scope *sc, FuncSig *sig, VecExprPtr args,
                           MLIR_ValueHandle *out_results) {
    if (args.size != sig->n_params) {
        println(str_lit("tinyc emit: call to {} arity mismatch"), sig->name);
    }
    size_t n_in = sig->n_flat_in;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, n_in ? n_in : 1);
    for (size_t i = 0; i < sig->n_params && i < args.size; i++) {
        SlotInfo *p = &sig->params[i];
        if (p->type.kind == TY_STRUCT) {
            flatten_struct_arg(e, sc, args.data[i], p, ops, p->flat_offset);
        } else {
            EVal v = emit_expr(e, sc, args.data[i]);
            ops[p->flat_offset] = coerce_eval(e, v, scalar_mlir_type(e, p->type.kind));
        }
    }
    size_t n_out = sig->n_flat_out;
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, n_out ? n_out : 1);
    MLIR_ValueHandle *rs  = arena_new_array(e->arena, MLIR_ValueHandle, n_out ? n_out : 1);
    for (size_t i = 0; i < n_out; i++) {
        rts[i] = sig->flat_out_tys[i];
        rs[i]  = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, (uint32_t)i,
                                          rts[i], ssa_name(e), eloc(e, 0));
        if (out_results) out_results[i] = rs[i];
    }
    MLIR_AttributeHandle callee_attr = MLIR_CreateAttributeSymbolRef(
        e->ctx, str_lit("callee"), sig->name);
    MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1);
    as[0] = callee_attr;
    emit_op(e, OP_TYPE_FUNC_CALL, str_lit("func.call"),
            rts, n_out, rs, n_out, ops, n_in, as, 1, NULL, 0);
}

static EVal emit_expr(E *e, Scope *sc, Expr *ex) {
    EVal r = {0};
    switch (ex->kind) {
        case EX_INT:
            r.val = emit_const_i32(e, ex->int_value); return r;
        case EX_FLOAT:
            r.val = emit_const_f32(e, ex->float_value); r.is_float = true; return r;
        case EX_VAR: {
            LVal lv = emit_lvalue(e, sc, ex);
            return load_lvalue(e, lv);
        }
        case EX_INDEX: {
            LVal lv = emit_lvalue(e, sc, ex);
            return load_lvalue(e, lv);
        }
        case EX_ADDR: {
            // &x where x is a local: the address IS the memref slot.
            // Result type is memref<i32> — we hand back a value whose
            // EVal "is_float" flag is meaningless because address-of can
            // only feed `*p` patterns or direct pointer assignment.
            if (ex->lhs->kind != EX_VAR) {
                println(str_lit("tinyc emit: &expr requires a simple variable"));
                return r;
            }
            Sym *s = scope_lookup(sc, ex->lhs->name);
            if (!s || s->type.kind != TY_I32) {
                println(str_lit("tinyc emit: &expr requires an int variable"));
                return r;
            }
            r.val = s->addr;
            return r;
        }
        case EX_DEREF: {
            LVal lv = emit_lvalue(e, sc, ex);
            return load_lvalue(e, lv);
        }
        case EX_FIELD: {
            LVal lv = emit_lvalue(e, sc, ex);
            return load_lvalue(e, lv);
        }
        case EX_ASSIGN: {
            // Struct rhs from a struct-returning call: q = f(p);
            if (ex->lvalue->kind == EX_VAR && ex->rhs_assign->kind == EX_CALL) {
                Sym *target = scope_lookup(sc, ex->lvalue->name);
                FuncSig *sig = find_sig(e, ex->rhs_assign->callee);
                if (target && target->type.kind == TY_STRUCT && target->sdef &&
                    sig && sig->ret.type.kind == TY_STRUCT &&
                    sig->ret.sdef == target->sdef) {
                    size_t n = sig->n_flat_out;
                    MLIR_ValueHandle *results = arena_new_array(e->arena, MLIR_ValueHandle,
                                                                n ? n : 1);
                    emit_flat_call(e, sc, sig, ex->rhs_assign->args, results);
                    for (size_t k = 0; k < n; k++) {
                        emit_store(e, results[k], target->field_addrs[k], NULL, 0);
                    }
                    return r;
                }
            }
            LVal lv = emit_lvalue(e, sc, ex->lvalue);
            EVal v = emit_expr(e, sc, ex->rhs_assign);
            v.val = coerce_eval(e, v, lv.elem_ty);
            v.is_float = (lv.elem_ty == e->f32);
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
            // OP_NOT: int-only (a == 0 ? 1 : 0)
            MLIR_ValueHandle b = emit_to_bool_i1(e, a);
            r.val = emit_select(e, e->i32, b, emit_const_i32(e, 0), emit_const_i32(e, 1));
            return r;
        }
        case EX_BIN: {
            // True short-circuit && / || via scf.if with i32 result.
            // Operands are coerced to int truthiness; the result is i32 0/1.
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
            unify_numeric(e, &a, &b);
            bool flt = a.is_float;
            switch (ex->op) {
                case OP_ADD:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_ADDF, str_lit("arith.addf"), e->f32, a.val, b.val)
                                : emit_binop(e, OP_TYPE_ARITH_ADDI, str_lit("arith.addi"), e->i32, a.val, b.val);
                    r.is_float = flt; return r;
                case OP_SUB:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_SUBF, str_lit("arith.subf"), e->f32, a.val, b.val)
                                : emit_binop(e, OP_TYPE_ARITH_SUBI, str_lit("arith.subi"), e->i32, a.val, b.val);
                    r.is_float = flt; return r;
                case OP_MUL:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_MULF, str_lit("arith.mulf"), e->f32, a.val, b.val)
                                : emit_binop(e, OP_TYPE_ARITH_MULI, str_lit("arith.muli"), e->i32, a.val, b.val);
                    r.is_float = flt; return r;
                case OP_DIV:
                    r.val = flt ? emit_binop(e, OP_TYPE_ARITH_DIVF, str_lit("arith.divf"), e->f32, a.val, b.val)
                                : emit_binop(e, OP_TYPE_ARITH_DIVSI, str_lit("arith.divsi"), e->i32, a.val, b.val);
                    r.is_float = flt; return r;
                case OP_MOD:
                    if (flt) {
                        println(str_lit("tinyc emit: % not supported on floats"));
                        r.val = emit_const_f32(e, 0.0); r.is_float = true; return r;
                    }
                    r.val = emit_binop(e, OP_TYPE_ARITH_REMSI, str_lit("arith.remsi"), e->i32, a.val, b.val);
                    return r;
                case OP_LT: case OP_LE: case OP_GT: case OP_GE:
                case OP_EQ: case OP_NE: {
                    MLIR_ValueHandle cmp = flt
                        ? emit_cmpf(e, cmpf_pred_for(ex->op), a.val, b.val)
                        : emit_cmpi(e, cmpi_pred_for(ex->op), a.val, b.val);
                    r.val = bool_to_i32(e, cmp);
                    return r;
                }
                default: break;
            }
            r.val = emit_const_i32(e, 0);
            return r;
        }
        case EX_CALL: {
            FuncSig *sig = find_sig(e, ex->callee);
            if (!sig) {
                println(str_lit("tinyc emit: unknown function {}"), ex->callee);
                r.val = emit_const_i32(e, 0);
                return r;
            }
            if (sig->ret.type.kind == TY_STRUCT) {
                // Struct return is only valid as the rhs of an assignment to
                // a struct lvalue, as a `return` expression in a struct-
                // returning function, or as a discarded ST_EXPR (handled at
                // the statement level). In an arbitrary expression position
                // we have nowhere to put the multiple results.
                println(str_lit("tinyc emit: struct-returning call cannot appear in expression position"));
                r.val = emit_const_i32(e, 0);
                return r;
            }
            MLIR_ValueHandle *results = arena_new_array(e->arena, MLIR_ValueHandle, 1);
            emit_flat_call(e, sc, sig, ex->args, results);
            r.val = results[0];
            r.is_float = (sig->ret.type.kind == TY_F32);
            return r;
        }
    }
    r.val = emit_const_i32(e, 0);
    return r;
}

static void emit_block(E *e, Scope *sc, VecStmtPtr body);

static void emit_stmt(E *e, Scope *sc, Stmt *st) {
    if (e->terminated) return;
    switch (st->kind) {
        case ST_EXPR: {
            // Detect `f(...);` where f returns struct (discard the result).
            if (st->expr->kind == EX_CALL) {
                FuncSig *sig = find_sig(e, st->expr->callee);
                if (sig && sig->ret.type.kind == TY_STRUCT) {
                    emit_flat_call(e, sc, sig, st->expr->args, NULL);
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

            if (st->decl_type.kind == TY_PTR_I32) {
                // Alias-only pointer: the decl MUST have an initializer of
                // the form &<var>; the pointer's "address" is just the
                // pointee's memref handle.
                if (!st->decl_init || st->decl_init->kind != EX_ADDR ||
                    st->decl_init->lhs->kind != EX_VAR) {
                    println(str_lit("tinyc emit: int* must be initialized with &<var>"));
                    sy->addr = emit_alloc(e, e->memref_i32);
                } else {
                    Sym *target = scope_lookup(sc, st->decl_init->lhs->name);
                    if (!target || target->type.kind != TY_I32) {
                        println(str_lit("tinyc emit: pointer target must be an int local"));
                        sy->addr = emit_alloc(e, e->memref_i32);
                    } else {
                        sy->addr = target->addr;
                    }
                }
            } else if (st->decl_type.kind == TY_ARRAY_I32) {
                int64_t shape[1] = { st->decl_type.array_len };
                MLIR_TypeHandle arr_ty = MLIR_CreateTypeMemref(e->ctx, shape, 1, e->i32);
                sy->addr = emit_alloc(e, arr_ty);
                if (st->decl_init) {
                    println(str_lit("tinyc emit: array initializers are not supported"));
                }
            } else if (st->decl_type.kind == TY_STRUCT) {
                StructDef *sd = find_struct(e, st->decl_type.struct_name);
                if (!sd) {
                    println(str_lit("tinyc emit: unknown struct type {}"), st->decl_type.struct_name);
                    return;
                }
                sy->sdef = sd;
                sy->field_addrs = arena_new_array(e->arena, MLIR_ValueHandle,
                                                  sd->fields.size ? sd->fields.size : 1);
                for (size_t i = 0; i < sd->fields.size; i++) {
                    bool flt = (sd->fields.data[i].kind == TY_F32);
                    MLIR_ValueHandle addr = emit_alloc(e, flt ? e->memref_f32 : e->memref_i32);
                    sy->field_addrs[i] = addr;
                    // zero-initialize each field
                    if (flt) emit_store(e, emit_const_f32(e, 0.0), addr, NULL, 0);
                    else     emit_store(e, emit_const_i32(e, 0),   addr, NULL, 0);
                }
                if (st->decl_init) {
                    println(str_lit("tinyc emit: struct initializers are not supported"));
                }
            } else if (st->decl_type.kind == TY_F32) {
                sy->addr = emit_alloc(e, e->memref_f32);
                if (st->decl_init) {
                    EVal v = emit_expr(e, sc, st->decl_init);
                    if (!v.is_float) { v.val = emit_sitofp(e, v.val); v.is_float = true; }
                    emit_store(e, v.val, sy->addr, NULL, 0);
                } else {
                    emit_store(e, emit_const_f32(e, 0.0), sy->addr, NULL, 0);
                }
            } else {
                sy->addr = emit_alloc(e, e->memref_i32);
                if (st->decl_init) {
                    MLIR_ValueHandle v = emit_expr_i32(e, sc, st->decl_init);
                    emit_store(e, v, sy->addr, NULL, 0);
                } else {
                    emit_store(e, emit_const_i32(e, 0), sy->addr, NULL, 0);
                }
            }
            sy->next = sc->head;
            sc->head = sy;
            return;
        }
        case ST_RETURN: {
            FuncSig *sig = e->cur_sig;
            // struct return: load each field and emit multi-result return.
            if (sig && sig->ret.type.kind == TY_STRUCT) {
                StructDef *want = sig->ret.sdef;
                size_t n = want ? want->fields.size : 0;
                MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, n ? n : 1);
                if (st->expr->kind == EX_CALL) {
                    // `return f(p);` where f also returns the same struct.
                    FuncSig *csig = find_sig(e, st->expr->callee);
                    if (!csig || csig->ret.type.kind != TY_STRUCT || csig->ret.sdef != want) {
                        println(str_lit("tinyc emit: returned call type mismatch"));
                        for (size_t k = 0; k < n; k++) ops[k] = emit_const_i32(e, 0);
                    } else {
                        emit_flat_call(e, sc, csig, st->expr->args, ops);
                    }
                } else if (st->expr->kind == EX_VAR) {
                    Sym *s = scope_lookup(sc, st->expr->name);
                    if (!s || s->type.kind != TY_STRUCT || s->sdef != want) {
                        println(str_lit("tinyc emit: returned struct type mismatch"));
                        for (size_t k = 0; k < n; k++) ops[k] = emit_const_i32(e, 0);
                    } else {
                        for (size_t k = 0; k < n; k++) {
                            bool flt = (s->sdef->fields.data[k].kind == TY_F32);
                            ops[k] = emit_load(e, s->field_addrs[k], flt ? e->f32 : e->i32, NULL, 0);
                        }
                    }
                } else {
                    println(str_lit("tinyc emit: struct return must be a variable or struct call"));
                    for (size_t k = 0; k < n; k++) ops[k] = emit_const_i32(e, 0);
                }
                emit_op(e, OP_TYPE_FUNC_RETURN, str_lit("func.return"),
                        NULL, 0, NULL, 0, ops, n, NULL, 0, NULL, 0);
                e->terminated = true;
                return;
            }
            // scalar return: coerce to declared return type.
            MLIR_TypeHandle want_ty = (sig && sig->ret.type.kind == TY_F32) ? e->f32 : e->i32;
            EVal v = emit_expr(e, sc, st->expr);
            MLIR_ValueHandle ret_v = coerce_eval(e, v, want_ty);
            MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = ret_v;
            emit_op(e, OP_TYPE_FUNC_RETURN, str_lit("func.return"),
                    NULL, 0, NULL, 0, ops, 1, NULL, 0, NULL, 0);
            e->terminated = true;
            return;
        }
        case ST_BREAK: {
            if (!e->loops) {
                println(str_lit("tinyc emit: break outside of loop"));
                return;
            }
            emit_branch(e, e->loops->break_block);
            return;
        }
        case ST_CONTINUE: {
            if (!e->loops) {
                println(str_lit("tinyc emit: continue outside of loop"));
                return;
            }
            emit_branch(e, e->loops->continue_block);
            return;
        }
        case ST_PRINT: {
            EVal v = emit_expr(e, sc, st->expr);
            // vector.print works on i32 and f32 scalars; the lowering pass
            // synthesizes the right runtime call (printI32/printF32) plus
            // printNewline automatically.
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

            // then
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

            // header: evaluate cond, cond_br
            e->cur_block = hdr_b; e->terminated = false;
            EVal c = emit_expr(e, sc, st->cond);
            MLIR_ValueHandle cb = emit_to_bool_i1(e, c);
            emit_cond_branch(e, cb, body_b, exit_b);

            // body
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
        case ST_FOR: {
            // for (init; cond; step) body  ->
            //   init;
            //   br hdr;
            // hdr: c = cond; cond_br c, body, exit
            // body: ... ; br step
            // step: step; br hdr
            // exit:
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
    }
}

static void emit_block(E *e, Scope *sc, VecStmtPtr body) {
    for (size_t i = 0; i < body.size && !e->terminated; i++) emit_stmt(e, sc, body.data[i]);
}

// ----- function & module -----

// Resolve a parser-level Type into a SlotInfo (struct ref + flat layout).
// Sets sdef, flat_count from struct definition (or 1 for scalar). flat_offset
// is filled in by the caller after layout pass.
static void slot_resolve(E *e, Type ty, SlotInfo *out) {
    out->type = ty;
    out->sdef = NULL;
    out->flat_count = 1;
    out->flat_offset = 0;
    if (ty.kind == TY_STRUCT) {
        out->sdef = find_struct(e, ty.struct_name);
        if (!out->sdef) {
            println(str_lit("tinyc emit: unknown struct type {}"), ty.struct_name);
            out->flat_count = 0;
        } else {
            out->flat_count = out->sdef->fields.size;
        }
    } else if (ty.kind != TY_I32 && ty.kind != TY_F32) {
        println(str_lit("tinyc emit: unsupported type in function signature"));
        out->flat_count = 0;
    }
}

// Append the flattened MLIR types of a slot into `tys` starting at `*offset`.
static void slot_emit_flat_types(E *e, SlotInfo *s, MLIR_TypeHandle *tys, size_t *offset) {
    s->flat_offset = *offset;
    if (s->type.kind == TY_STRUCT) {
        if (!s->sdef) return;
        for (size_t i = 0; i < s->sdef->fields.size; i++) {
            tys[(*offset)++] = scalar_mlir_type(e, s->sdef->fields.data[i].kind);
        }
    } else {
        tys[(*offset)++] = scalar_mlir_type(e, s->type.kind);
    }
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
        slot_resolve(e, f->return_type, &sig->ret);
        sig->n_params = f->params.size;
        sig->params = arena_new_array(e->arena, SlotInfo, sig->n_params ? sig->n_params : 1);
        size_t in_total = 0;
        for (size_t i = 0; i < sig->n_params; i++) {
            slot_resolve(e, f->params.data[i].type, &sig->params[i]);
            in_total += sig->params[i].flat_count;
        }
        size_t out_total = sig->ret.flat_count;
        sig->flat_in_tys  = arena_new_array(e->arena, MLIR_TypeHandle, in_total ? in_total : 1);
        sig->flat_out_tys = arena_new_array(e->arena, MLIR_TypeHandle, out_total ? out_total : 1);
        size_t off = 0;
        for (size_t i = 0; i < sig->n_params; i++) {
            slot_emit_flat_types(e, &sig->params[i], sig->flat_in_tys, &off);
        }
        sig->n_flat_in = off;
        off = 0;
        slot_emit_flat_types(e, &sig->ret, sig->flat_out_tys, &off);
        sig->n_flat_out = off;
        sig->fn_ty = MLIR_CreateTypeFunction(e->ctx,
            sig->flat_in_tys, sig->n_flat_in, sig->flat_out_tys, sig->n_flat_out);
    }
}

static MLIR_OpHandle emit_func(E *e, Func *f) {
    FuncSig *sig = find_sig(e, f->name);

    MLIR_RegionHandle body_r = MLIR_CreateRegion(e->ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(e->ctx);
    MLIR_AppendRegionBlock(e->ctx, body_r, entry);

    Scope sc = (Scope){.head = NULL, .parent = NULL};
    // Create one BlockArg per flattened scalar parameter.
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
    e->cur_block = entry;
    e->func_region = body_r;
    e->cur_sig = sig;
    e->terminated = false;
    e->next_ssa = 0;

    // Spill each parameter into per-field memrefs (struct) or one
    // memref (scalar) so the body can mutate it like any other local.
    for (size_t i = 0; i < sig->n_params; i++) {
        SlotInfo *p = &sig->params[i];
        Sym *sy = arena_new(e->arena, Sym);
        sy->name = f->params.data[i].name;
        sy->type = p->type;
        if (p->type.kind == TY_STRUCT && p->sdef) {
            sy->sdef = p->sdef;
            sy->field_addrs = arena_new_array(e->arena, MLIR_ValueHandle,
                                              p->sdef->fields.size ? p->sdef->fields.size : 1);
            for (size_t k = 0; k < p->sdef->fields.size; k++) {
                bool flt = (p->sdef->fields.data[k].kind == TY_F32);
                MLIR_ValueHandle addr = emit_alloc(e, flt ? e->memref_f32 : e->memref_i32);
                sy->field_addrs[k] = addr;
                emit_store(e, flat_args[p->flat_offset + k], addr, NULL, 0);
            }
        } else {
            bool flt = (p->type.kind == TY_F32);
            MLIR_ValueHandle addr = emit_alloc(e, flt ? e->memref_f32 : e->memref_i32);
            emit_store(e, flat_args[p->flat_offset], addr, NULL, 0);
            sy->addr = addr;
        }
        sy->next = sc.head;
        sc.head = sy;
    }

    emit_block(e, &sc, f->body);
    // Fall-through safety: if the user's last block did not terminate,
    // emit a default return so the IR is structurally valid.
    if (!e->terminated) {
        MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle,
                                                sig->n_flat_out ? sig->n_flat_out : 1);
        for (size_t i = 0; i < sig->n_flat_out; i++) {
            MLIR_TypeHandle ty = sig->flat_out_tys[i];
            ops[i] = (ty == e->f32) ? emit_const_f32(e, 0.0) : emit_const_i32(e, 0);
        }
        emit_op(e, OP_TYPE_FUNC_RETURN, str_lit("func.return"),
                NULL, 0, NULL, 0, ops, sig->n_flat_out, NULL, 0, NULL, 0);
        e->terminated = true;
    }
    e->cur_block = saved;
    e->func_region = saved_region;
    e->cur_sig = saved_sig;

    MLIR_AttributeHandle sym_name = MLIR_CreateAttributeString(e->ctx, str_lit("sym_name"), f->name);
    MLIR_AttributeHandle fn_ty_attr = MLIR_CreateAttributeType(e->ctx, str_lit("function_type"), sig->fn_ty);
    MLIR_AttributeHandle *attrs = arena_new_array(e->arena, MLIR_AttributeHandle, 2);
    attrs[0] = sym_name; attrs[1] = fn_ty_attr;
    MLIR_RegionHandle *regs = arena_new_array(e->arena, MLIR_RegionHandle, 1); regs[0] = body_r;
    MLIR_OpHandle fn = MLIR_CreateOp(e->ctx, OP_TYPE_FUNC_FUNC, str_lit("func.func"),
                                     attrs, 2, NULL, 0, NULL, 0, NULL, 0,
                                     regs, 1, eloc(e, 0), MLIR_INVALID_HANDLE,
                                     str_lit(""), -1);
    return fn;
}

MLIR_OpHandle tinyc_emit_module(MLIR_Context *ctx, Program *program) {
    Arena *arena = MLIR_GetArenaAllocator(ctx);
    E e = (E){0};
    e.ctx = ctx;
    e.arena = arena;
    e.i32 = MLIR_CreateTypeInteger(ctx, 32, true);
    e.i1  = MLIR_CreateTypeInteger(ctx, 1, false);
    e.f32 = MLIR_CreateTypeFloat(ctx, 32, false);
    e.index = MLIR_CreateTypeIndex(ctx);
    int64_t shape0[1] = {0};   // 0 dims: memref<i32> / memref<f32>
    e.memref_i32 = MLIR_CreateTypeMemref(ctx, shape0, 0, e.i32);
    e.memref_f32 = MLIR_CreateTypeMemref(ctx, shape0, 0, e.f32);
    e.loc = MLIR_CreateLocationUnknown(ctx, str_lit(""));
    e.next_ssa = 0;
    e.cur_block = MLIR_INVALID_HANDLE;
    e.func_region = MLIR_INVALID_HANDLE;
    e.terminated = false;
    e.loops = NULL;
    e.program = program;
    e.cur_sig = NULL;

    build_signatures(&e);

    // builtin.module { ... }
    MLIR_RegionHandle mr = MLIR_CreateRegion(ctx);
    MLIR_BlockHandle mb = MLIR_CreateBlock(ctx);
    MLIR_AppendRegionBlock(ctx, mr, mb);
    MLIR_RegionHandle *mregs = arena_new_array(arena, MLIR_RegionHandle, 1); mregs[0] = mr;
    MLIR_OpHandle module = MLIR_CreateOp(ctx, OP_TYPE_MODULE, str_lit("module"),
                                         NULL, 0, NULL, 0, NULL, 0, NULL, 0,
                                         mregs, 1, e.loc, MLIR_INVALID_HANDLE,
                                         str_lit(""), -1);

    for (size_t i = 0; i < program->funcs.size; i++) {
        MLIR_OpHandle fn = emit_func(&e, program->funcs.data[i]);
        MLIR_AppendBlockOp(ctx, mb, fn);
    }
    return module;
}

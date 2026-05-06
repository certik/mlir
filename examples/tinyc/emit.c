// tinyC -> MLIR emitter. Uses ONLY mlir_api.h (plus corec primitives).
// No upstream MLIR headers are included from this directory.

#include "tinyc.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <base/arena.h>
#include <base/format.h>
#include <base/io.h>
#include <base/string.h>

// ----- small symbol table (linear scan; tinyC programs are small) -----

typedef struct Sym {
    string name;
    MLIR_ValueHandle addr;     // memref<i32> handle
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

typedef struct {
    MLIR_Context *ctx;
    Arena *arena;
    MLIR_TypeHandle i32;
    MLIR_TypeHandle i1;
    MLIR_TypeHandle memref_i32;       // memref<i32>
    MLIR_BlockHandle cur_block;       // where the next op gets appended
    MLIR_LocationHandle loc;
    int next_ssa;                     // running %N counter
} E;

static MLIR_LocationHandle eloc(E *e, int line) {
    (void)line;
    return e->loc;
}

static string ssa_name(E *e) {
    int n = e->next_ssa++;
    return format(e->arena, str_lit("%{}"), (int64_t)n);
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

static MLIR_ValueHandle emit_alloc_local(E *e) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->memref_i32, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->memref_i32;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    emit_op(e, OP_TYPE_MEMREF_ALLOC, str_lit("memref.alloc"),
            rts, 1, rs, 1, NULL, 0, NULL, 0, NULL, 0);
    return r;
}

static void emit_store(E *e, MLIR_ValueHandle val, MLIR_ValueHandle addr) {
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 2);
    ops[0] = val; ops[1] = addr;
    emit_op(e, OP_TYPE_MEMREF_STORE, str_lit("memref.store"),
            NULL, 0, NULL, 0, ops, 2, NULL, 0, NULL, 0);
}

static MLIR_ValueHandle emit_load(E *e, MLIR_ValueHandle addr) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i32, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i32;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = addr;
    emit_op(e, OP_TYPE_MEMREF_LOAD, str_lit("memref.load"),
            rts, 1, rs, 1, ops, 1, NULL, 0, NULL, 0);
    return r;
}

static MLIR_ValueHandle emit_select(E *e, MLIR_ValueHandle c,
                                    MLIR_ValueHandle t, MLIR_ValueHandle f) {
    MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                  e->i32, ssa_name(e), eloc(e, 0));
    MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i32;
    MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
    MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 3);
    ops[0] = c; ops[1] = t; ops[2] = f;
    emit_op(e, OP_TYPE_ARITH_SELECT, str_lit("arith.select"),
            rts, 1, rs, 1, ops, 3, NULL, 0, NULL, 0);
    return r;
}

// MLIR cmpi predicate codes (see arith dialect):
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

// ----- expression / statement emission -----

static MLIR_ValueHandle emit_expr(E *e, Scope *sc, Expr *ex);

static MLIR_ValueHandle emit_to_bool(E *e, MLIR_ValueHandle v) {
    // i32 -> i1 by `v != 0`
    MLIR_ValueHandle z = emit_const_i32(e, 0);
    return emit_cmpi(e, /*ne*/ 1, v, z);
}

static MLIR_ValueHandle bool_to_i32(E *e, MLIR_ValueHandle b) {
    MLIR_ValueHandle one  = emit_const_i32(e, 1);
    MLIR_ValueHandle zero = emit_const_i32(e, 0);
    return emit_select(e, b, one, zero);
}

static MLIR_ValueHandle emit_expr(E *e, Scope *sc, Expr *ex) {
    switch (ex->kind) {
        case EX_INT:
            return emit_const_i32(e, ex->int_value);
        case EX_VAR: {
            Sym *s = scope_lookup(sc, ex->name);
            if (!s) {
                println(str_lit("tinyc emit: undefined variable {}"), ex->name);
                return emit_const_i32(e, 0);
            }
            return emit_load(e, s->addr);
        }
        case EX_ASSIGN: {
            Sym *s = scope_lookup(sc, ex->name);
            if (!s) {
                println(str_lit("tinyc emit: undefined variable {}"), ex->name);
                return emit_const_i32(e, 0);
            }
            MLIR_ValueHandle v = emit_expr(e, sc, ex->rhs_assign);
            emit_store(e, v, s->addr);
            return v;
        }
        case EX_UN: {
            MLIR_ValueHandle a = emit_expr(e, sc, ex->lhs);
            if (ex->op == OP_NEG) {
                MLIR_ValueHandle z = emit_const_i32(e, 0);
                return emit_binop(e, OP_TYPE_ARITH_SUBI, str_lit("arith.subi"), e->i32, z, a);
            }
            // OP_NOT: (a == 0) ? 1 : 0
            MLIR_ValueHandle z = emit_const_i32(e, 0);
            MLIR_ValueHandle eq = emit_cmpi(e, /*eq*/ 0, a, z);
            return bool_to_i32(e, eq);
        }
        case EX_BIN: {
            // Short-circuiting && and || via select (not strictly short-circuit
            // semantically but tinyC has no side-effects in pure exprs).
            if (ex->op == OP_AND || ex->op == OP_OR) {
                MLIR_ValueHandle a = emit_expr(e, sc, ex->lhs);
                MLIR_ValueHandle b = emit_expr(e, sc, ex->rhs);
                MLIR_ValueHandle ab = emit_to_bool(e, a);
                MLIR_ValueHandle bb = emit_to_bool(e, b);
                MLIR_ValueHandle r = (ex->op == OP_AND)
                    ? emit_binop(e, OP_TYPE_ARITH_ANDI, str_lit("arith.andi"), e->i1, ab, bb)
                    : emit_binop(e, OP_TYPE_ARITH_ORI,  str_lit("arith.ori"),  e->i1, ab, bb);
                return bool_to_i32(e, r);
            }
            MLIR_ValueHandle a = emit_expr(e, sc, ex->lhs);
            MLIR_ValueHandle b = emit_expr(e, sc, ex->rhs);
            switch (ex->op) {
                case OP_ADD: return emit_binop(e, OP_TYPE_ARITH_ADDI,  str_lit("arith.addi"),  e->i32, a, b);
                case OP_SUB: return emit_binop(e, OP_TYPE_ARITH_SUBI,  str_lit("arith.subi"),  e->i32, a, b);
                case OP_MUL: return emit_binop(e, OP_TYPE_ARITH_MULI,  str_lit("arith.muli"),  e->i32, a, b);
                case OP_DIV: return emit_binop(e, OP_TYPE_ARITH_DIVSI, str_lit("arith.divsi"), e->i32, a, b);
                case OP_MOD: return emit_binop(e, OP_TYPE_ARITH_REMSI, str_lit("arith.remsi"), e->i32, a, b);
                case OP_LT: case OP_LE: case OP_GT: case OP_GE:
                case OP_EQ: case OP_NE: {
                    MLIR_ValueHandle r = emit_cmpi(e, cmpi_pred_for(ex->op), a, b);
                    return bool_to_i32(e, r);
                }
                default: break;
            }
            return emit_const_i32(e, 0);
        }
        case EX_CALL: {
            // Build operand list
            size_t n = ex->args.size;
            MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, n ? n : 1);
            for (size_t i = 0; i < n; i++) {
                ops[i] = emit_expr(e, sc, ex->args.data[i]);
            }
            MLIR_ValueHandle r = MLIR_CreateValueOpResult(e->ctx, MLIR_INVALID_HANDLE, 0,
                                                          e->i32, ssa_name(e), eloc(e, 0));
            MLIR_TypeHandle *rts = arena_new_array(e->arena, MLIR_TypeHandle, 1); rts[0] = e->i32;
            MLIR_ValueHandle *rs = arena_new_array(e->arena, MLIR_ValueHandle, 1); rs[0] = r;
            MLIR_AttributeHandle callee_attr = MLIR_CreateAttributeSymbolRef(
                e->ctx, str_lit("callee"), ex->callee);
            MLIR_AttributeHandle *as = arena_new_array(e->arena, MLIR_AttributeHandle, 1);
            as[0] = callee_attr;
            emit_op(e, OP_TYPE_FUNC_CALL, str_lit("func.call"),
                    rts, 1, rs, 1, ops, n, as, 1, NULL, 0);
            return r;
        }
    }
    return emit_const_i32(e, 0);
}

static void emit_block(E *e, Scope *parent, VecStmtPtr body);

static void emit_stmt(E *e, Scope *sc, Stmt *st) {
    switch (st->kind) {
        case ST_EXPR:
            (void)emit_expr(e, sc, st->expr);
            return;
        case ST_DECL: {
            MLIR_ValueHandle addr = emit_alloc_local(e);
            Sym *sy = arena_new(e->arena, Sym);
            sy->name = st->decl_name;
            sy->addr = addr;
            sy->next = sc->head;
            sc->head = sy;
            if (st->decl_init) {
                MLIR_ValueHandle v = emit_expr(e, sc, st->decl_init);
                emit_store(e, v, addr);
            } else {
                emit_store(e, emit_const_i32(e, 0), addr);
            }
            return;
        }
        case ST_RETURN: {
            MLIR_ValueHandle v = emit_expr(e, sc, st->expr);
            MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v;
            emit_op(e, OP_TYPE_FUNC_RETURN, str_lit("func.return"),
                    NULL, 0, NULL, 0, ops, 1, NULL, 0, NULL, 0);
            return;
        }
        case ST_PRINT: {
            MLIR_ValueHandle v = emit_expr(e, sc, st->expr);
            MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = v;
            emit_op(e, OP_TYPE_VECTOR_PRINT, str_lit("vector.print"),
                    NULL, 0, NULL, 0, ops, 1, NULL, 0, NULL, 0);
            return;
        }
        case ST_BLOCK: {
            Scope inner = (Scope){.head = NULL, .parent = sc};
            emit_block(e, &inner, st->block_body);
            (void)inner;
            // Outer scope is restored implicitly; new sub-scope doesn't leak.
            // (Sym entries live in arena and are simply unreachable now.)
            // To keep symbols visible in the same flat function scope, we
            // emit_block with `sc` instead. Here we choose lexical scoping.
            return;
        }
        case ST_IF: {
            MLIR_ValueHandle c = emit_to_bool(e, emit_expr(e, sc, st->cond));
            MLIR_RegionHandle then_r = MLIR_CreateRegion(e->ctx);
            MLIR_BlockHandle then_b = MLIR_CreateBlock(e->ctx);
            MLIR_AppendRegionBlock(e->ctx, then_r, then_b);

            MLIR_RegionHandle else_r = MLIR_CreateRegion(e->ctx);
            MLIR_BlockHandle else_b = MLIR_CreateBlock(e->ctx);
            MLIR_AppendRegionBlock(e->ctx, else_r, else_b);

            MLIR_RegionHandle *regs = arena_new_array(e->arena, MLIR_RegionHandle, 2);
            regs[0] = then_r; regs[1] = else_r;
            MLIR_ValueHandle *ops = arena_new_array(e->arena, MLIR_ValueHandle, 1); ops[0] = c;
            MLIR_OpHandle if_op = emit_op(e, OP_TYPE_SCF_IF, str_lit("scf.if"),
                                          NULL, 0, NULL, 0, ops, 1, NULL, 0, regs, 2);
            (void)if_op;

            MLIR_BlockHandle saved = e->cur_block;
            // then
            e->cur_block = then_b;
            { Scope inner = (Scope){.head = NULL, .parent = sc};
              for (size_t i = 0; i < st->then_body.size; i++) emit_stmt(e, &inner, st->then_body.data[i]); }
            emit_op(e, OP_TYPE_SCF_YIELD, str_lit("scf.yield"),
                    NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0);
            // else
            e->cur_block = else_b;
            if (st->has_else) {
                Scope inner = (Scope){.head = NULL, .parent = sc};
                for (size_t i = 0; i < st->else_body.size; i++) emit_stmt(e, &inner, st->else_body.data[i]);
            }
            emit_op(e, OP_TYPE_SCF_YIELD, str_lit("scf.yield"),
                    NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0);
            e->cur_block = saved;
            return;
        }
        case ST_WHILE: {
            // scf.while : () -> () {
            //   <before>: cond
            //   scf.condition(%c)
            // } do {
            //   <after>: body
            //   scf.yield
            // }
            MLIR_RegionHandle before_r = MLIR_CreateRegion(e->ctx);
            MLIR_BlockHandle before_b = MLIR_CreateBlock(e->ctx);
            MLIR_AppendRegionBlock(e->ctx, before_r, before_b);

            MLIR_RegionHandle after_r = MLIR_CreateRegion(e->ctx);
            MLIR_BlockHandle after_b = MLIR_CreateBlock(e->ctx);
            MLIR_AppendRegionBlock(e->ctx, after_r, after_b);

            MLIR_RegionHandle *regs = arena_new_array(e->arena, MLIR_RegionHandle, 2);
            regs[0] = before_r; regs[1] = after_r;
            emit_op(e, OP_TYPE_SCF_WHILE, str_lit("scf.while"),
                    NULL, 0, NULL, 0, NULL, 0, NULL, 0, regs, 2);

            MLIR_BlockHandle saved = e->cur_block;
            e->cur_block = before_b;
            MLIR_ValueHandle c = emit_to_bool(e, emit_expr(e, sc, st->cond));
            MLIR_ValueHandle *cops = arena_new_array(e->arena, MLIR_ValueHandle, 1); cops[0] = c;
            // scf.condition is currently OP_TYPE_UNREGISTERED in our enum;
            // emit by name. Generic form is acceptable here.
            emit_op(e, OP_TYPE_UNREGISTERED, str_lit("scf.condition"),
                    NULL, 0, NULL, 0, cops, 1, NULL, 0, NULL, 0);

            e->cur_block = after_b;
            { Scope inner = (Scope){.head = NULL, .parent = sc};
              for (size_t i = 0; i < st->while_body.size; i++) emit_stmt(e, &inner, st->while_body.data[i]); }
            emit_op(e, OP_TYPE_SCF_YIELD, str_lit("scf.yield"),
                    NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0);
            e->cur_block = saved;
            return;
        }
    }
}

static void emit_block(E *e, Scope *sc, VecStmtPtr body) {
    for (size_t i = 0; i < body.size; i++) emit_stmt(e, sc, body.data[i]);
}

// ----- function & module -----

static MLIR_OpHandle emit_func(E *e, Func *f) {
    // Build function type (i32, i32, ...) -> i32
    MLIR_TypeHandle *in_tys = arena_new_array(e->arena, MLIR_TypeHandle, f->params.size ? f->params.size : 1);
    for (size_t i = 0; i < f->params.size; i++) in_tys[i] = e->i32;
    MLIR_TypeHandle *out_tys = arena_new_array(e->arena, MLIR_TypeHandle, 1); out_tys[0] = e->i32;
    MLIR_TypeHandle fn_ty = MLIR_CreateTypeFunction(e->ctx, in_tys, f->params.size, out_tys, 1);

    MLIR_RegionHandle body_r = MLIR_CreateRegion(e->ctx);
    MLIR_BlockHandle entry = MLIR_CreateBlock(e->ctx);
    MLIR_AppendRegionBlock(e->ctx, body_r, entry);

    // Block args + symbol entries that copy them into local memref slots.
    Scope sc = (Scope){.head = NULL, .parent = NULL};
    MLIR_ValueHandle *param_args = arena_new_array(e->arena, MLIR_ValueHandle,
                                                   f->params.size ? f->params.size : 1);
    for (size_t i = 0; i < f->params.size; i++) {
        string nm = format(e->arena, str_lit("%arg{}"), (int64_t)i);
        param_args[i] = MLIR_CreateValueBlockArg(e->ctx, nm, (uint32_t)i, e->i32, eloc(e, 0));
        MLIR_AppendBlockArg(e->ctx, entry, param_args[i]);
    }

    MLIR_BlockHandle saved = e->cur_block;
    e->cur_block = entry;
    e->next_ssa = 0;

    // Spill each parameter into a memref so users can reassign it.
    for (size_t i = 0; i < f->params.size; i++) {
        MLIR_ValueHandle addr = emit_alloc_local(e);
        emit_store(e, param_args[i], addr);
        Sym *sy = arena_new(e->arena, Sym);
        sy->name = f->params.data[i].name;
        sy->addr = addr;
        sy->next = sc.head;
        sc.head = sy;
    }

    emit_block(e, &sc, f->body);
    e->cur_block = saved;

    // func.func op
    MLIR_AttributeHandle sym_name = MLIR_CreateAttributeString(e->ctx, str_lit("sym_name"), f->name);
    MLIR_AttributeHandle fn_ty_attr = MLIR_CreateAttributeType(e->ctx, str_lit("function_type"), fn_ty);
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
    int64_t shape[1] = {0};   // 0 dims: memref<i32>
    e.memref_i32 = MLIR_CreateTypeMemref(ctx, shape, 0, e.i32);
    e.loc = MLIR_CreateLocationUnknown(ctx, str_lit(""));
    e.next_ssa = 0;
    e.cur_block = MLIR_INVALID_HANDLE;

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

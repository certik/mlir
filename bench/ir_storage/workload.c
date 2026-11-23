// IR storage benchmark: the workload. It only talks to ir_api.h, so the same
// code drives every storage design.
//
//   ir_bench N REPS [HINT]
//
// builds a module of roughly N ops shaped like tinyc's LLVM-dialect output,
// then times four phases, REPS + 1 times (the first run is a warmup; each run
// starts from a fresh context):
//   build    create the module: functions, blocks, ops, use lists
//   walk     visit every op in every block: op type, operand count, each
//            operand's defining op and type, successor operands
//   uses     walk the use list of every value (block arguments and results)
//   rewrite  constant folding (create the constant, insert it, RAUW, erase the
//            folded op) followed by DCE to a fixed point
// With HINT = 1 the context is first given a size hint of 1.125 N ops
// (ir_ctx_reserve), outside the timed region. One line per run goes to
// stdout. The op count, rewrite counts and checksums must agree across
// designs.
#include "ir_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <platform/platform.h>
#include <platform/syscall6.h>

static uint64_t now_ns(void) {
    int64_t ts[2];
    __builtin_syscall6(228 /* clock_gettime */, 1 /* CLOCK_MONOTONIC */, (long)ts, 0, 0, 0, 0);
    return (uint64_t)ts[0] * 1000000000ull + (uint64_t)ts[1];
}

// --- deterministic generator ---

typedef struct Gen {
    IrCtx *c;
    uint32_t rng;
    IrType i1, i32, i64, ptr;
    IrBlock cur;
    uint64_t n_ops;
    IrValue ints[32];   // recent i32 values usable as operands
    uint32_t n_ints;
    IrValue pending[8];
    uint32_t n_pending;
    IrValue bools[8];
    uint32_t n_bools;
    IrValue ptrs[32];   // allocas of the function
    uint32_t n_allocas;
    uint32_t n_ptrs;
} Gen;

static uint32_t rnd(Gen *g) {
    uint32_t x = g->rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g->rng = x;
    return x;
}

static uint32_t rnd_n(Gen *g, uint32_t n) { return rnd(g) % n; }

static IrOp emit(Gen *g, uint32_t op_type, const IrValue *operands, uint32_t n_operands,
                 const IrType *result_types, uint32_t n_results,
                 const IrAttr *attrs, uint32_t n_attrs) {
    IrOpSpec s;
    s.op_type = op_type;
    s.operands = operands;
    s.n_operands = n_operands;
    s.result_types = result_types;
    s.n_results = n_results;
    s.attrs = attrs;
    s.n_attrs = n_attrs;
    s.regions = NULL;
    s.n_regions = 0;
    s.succs = NULL;
    s.n_succs = 0;
    s.n_succ_operands = NULL;
    IrOp op = ir_op_create(g->c, &s);
    ir_block_append_op(g->c, g->cur, op);
    g->n_ops++;
    return op;
}

// i32 operands: new values wait on a small stack and are consumed first, as
// in expression trees, so few values end up dead; otherwise an operand is one
// of the 32 most recent values.
static void push_int(Gen *g, IrValue v) {
    g->ints[g->n_ints % 32] = v;
    g->n_ints++;
    if (g->n_pending < 8) g->pending[g->n_pending++] = v;
}

static IrValue pick_int(Gen *g) {
    if (g->n_pending > 0 && rnd_n(g, 100) < 85) return g->pending[--g->n_pending];
    uint32_t n = g->n_ints < 32 ? g->n_ints : 32;
    return g->ints[(g->n_ints - 1 - rnd_n(g, n)) % 32];
}

static IrValue make_const(Gen *g, int64_t value, IrType t) {
    IrAttr a = ir_attr_int(g->c, ATTR_VALUE, value, t);
    IrOp op = emit(g, OPK_CONST, NULL, 0, &t, 1, &a, 1);
    return ir_op_result(g->c, op, 0);
}

static void gen_body_op(Gen *g) {
    IrCtx *c = g->c;
    uint32_t r = rnd_n(g, 100);
    IrValue ops[3];
    IrAttr at;
    if (r < 8) {                       // standalone constant
        push_int(g, make_const(g, (int64_t)rnd_n(g, 100), g->i32));
    } else if (r < 32) {               // constant index, getelementptr, load or store
        ops[0] = g->ptrs[rnd_n(g, g->n_ptrs)];
        ops[1] = make_const(g, (int64_t)rnd_n(g, 8), g->i64);
        at = ir_attr_int(c, ATTR_ELEM_TYPE, 32, g->i32);
        IrValue p = ir_op_result(c, emit(g, OPK_GEP, ops, 2, &g->ptr, 1, &at, 1), 0);
        if (rnd_n(g, 100) < 55) {
            push_int(g, ir_op_result(c, emit(g, OPK_LOAD, &p, 1, &g->i32, 1, NULL, 0), 0));
        } else {
            ops[0] = pick_int(g);
            ops[1] = p;
            emit(g, OPK_STORE, ops, 2, NULL, 0, NULL, 0);
        }
    } else if (r < 44) {               // load of a local
        ops[0] = g->ptrs[rnd_n(g, g->n_ptrs)];
        push_int(g, ir_op_result(c, emit(g, OPK_LOAD, ops, 1, &g->i32, 1, NULL, 0), 0));
    } else if (r < 56) {               // store to a local
        ops[0] = pick_int(g);
        ops[1] = g->ptrs[rnd_n(g, g->n_ptrs)];
        emit(g, OPK_STORE, ops, 2, NULL, 0, NULL, 0);
    } else if (r < 62) {               // icmp
        ops[0] = pick_int(g);
        ops[1] = pick_int(g);
        at = ir_attr_int(c, ATTR_PREDICATE, (int64_t)rnd_n(g, 4), g->i64);
        IrValue v = ir_op_result(c, emit(g, OPK_ICMP, ops, 2, &g->i1, 1, &at, 1), 0);
        if (g->n_bools < 8) g->bools[g->n_bools++] = v;
    } else if (r < 78) {               // add / sub / mul, a third of them foldable
        uint32_t k = r < 71 ? OPK_ADD : (r < 75 ? OPK_SUB : OPK_MUL);
        if (rnd_n(g, 3) == 0) {
            ops[0] = make_const(g, (int64_t)rnd_n(g, 50), g->i32);
            ops[1] = make_const(g, (int64_t)rnd_n(g, 50), g->i32);
        } else {
            ops[0] = pick_int(g);
            ops[1] = pick_int(g);
        }
        push_int(g, ir_op_result(c, emit(g, k, ops, 2, &g->i32, 1, NULL, 0), 0));
    } else if (r < 80) {               // sext, usually dead
        ops[0] = pick_int(g);
        emit(g, OPK_SEXT, ops, 1, &g->i64, 1, NULL, 0);
    } else if (r < 88 && g->n_bools > 0) {  // select
        ops[0] = g->bools[--g->n_bools];
        ops[1] = pick_int(g);
        ops[2] = pick_int(g);
        push_int(g, ir_op_result(c, emit(g, OPK_SELECT, ops, 3, &g->i32, 1, NULL, 0), 0));
    } else {                           // call
        ops[0] = pick_int(g);
        ops[1] = pick_int(g);
        at = ir_attr_int(c, ATTR_CALLEE, (int64_t)rnd_n(g, 64), g->i64);
        push_int(g, ir_op_result(c, emit(g, OPK_CALL, ops, 2, &g->i32, 1, &at, 1), 0));
    }
}

static void emit_branch(Gen *g, IrBlock *blocks, uint32_t n_blocks, uint32_t bi) {
    IrCtx *c = g->c;
    IrBlock succs[2];
    uint32_t n_succ_ops[2];
    IrValue operands[3];
    uint32_t n = 0;
    uint32_t n_succs = 1;
    succs[0] = blocks[bi + 1];
    if (g->n_bools > 0 && rnd_n(g, 100) < 55) {
        n_succs = 2;
        operands[n++] = g->bools[--g->n_bools];
        uint32_t t = bi + 2 + rnd_n(g, 3);
        succs[1] = blocks[t < n_blocks ? t : n_blocks - 1];
    }
    uint32_t n_regular = n;
    for (uint32_t s = 0; s < n_succs; s++) {
        n_succ_ops[s] = ir_block_num_args(c, succs[s]);
        for (uint32_t k = 0; k < n_succ_ops[s]; k++) operands[n++] = pick_int(g);
    }
    IrOpSpec sp;
    sp.op_type = n_succs == 2 ? OPK_COND_BR : OPK_BR;
    sp.operands = operands;
    sp.n_operands = n_regular;
    sp.result_types = NULL;
    sp.n_results = 0;
    sp.attrs = NULL;
    sp.n_attrs = 0;
    sp.regions = NULL;
    sp.n_regions = 0;
    sp.succs = succs;
    sp.n_succs = n_succs;
    sp.n_succ_operands = n_succ_ops;
    ir_block_append_op(c, g->cur, ir_op_create(c, &sp));
    g->n_ops++;
}

static void gen_func(Gen *g, IrBlock module_body, uint32_t fidx) {
    IrCtx *c = g->c;
    IrRegion region = ir_region_create(c);
    IrBlock blocks[32];
    uint32_t n_blocks = 6 + rnd_n(g, 20);
    for (uint32_t i = 0; i < n_blocks; i++) {
        blocks[i] = ir_block_create(c);
        ir_region_append_block(c, region, blocks[i]);
        if (i > 0 && i % 4 == 0) ir_block_add_arg(c, blocks[i], g->i32);
    }
    IrValue arg0 = ir_block_add_arg(c, blocks[0], g->i32);
    ir_block_add_arg(c, blocks[0], g->ptr);
    IrAttr fattrs[2];
    fattrs[0] = ir_attr_int(c, ATTR_SYM_NAME, (int64_t)fidx, g->i64);
    fattrs[1] = ir_attr_int(c, ATTR_FUNC_TYPE, 2, g->i64);
    IrOpSpec fs;
    fs.op_type = OPK_FUNC;
    fs.operands = NULL;
    fs.n_operands = 0;
    fs.result_types = NULL;
    fs.n_results = 0;
    fs.attrs = fattrs;
    fs.n_attrs = 2;
    fs.regions = &region;
    fs.n_regions = 1;
    fs.succs = NULL;
    fs.n_succs = 0;
    fs.n_succ_operands = NULL;
    ir_block_append_op(c, module_body, ir_op_create(c, &fs));
    g->n_ops++;

    // Entry block: allocas, each sized by a constant.
    g->cur = blocks[0];
    g->n_allocas = 0;
    uint32_t n_allocas = 3 + rnd_n(g, 6);
    for (uint32_t i = 0; i < n_allocas; i++) {
        IrValue one = make_const(g, 1, g->i64);
        IrOp a = emit(g, OPK_ALLOCA, &one, 1, &g->ptr, 1, NULL, 0);
        g->ptrs[g->n_allocas++] = ir_op_result(c, a, 0);
    }
    for (uint32_t bi = 0; bi < n_blocks; bi++) {
        g->cur = blocks[bi];
        g->n_ptrs = g->n_allocas;
        g->n_ints = 0;
        g->n_pending = 0;
        g->n_bools = 0;
        push_int(g, arg0);
        uint32_t na = ir_block_num_args(c, blocks[bi]);
        for (uint32_t k = 0; k < na; k++) {
            IrValue a = ir_block_arg(c, blocks[bi], k);
            if (ir_value_type(c, a) == g->i32) push_int(g, a);
        }
        uint32_t n_body = 4 + rnd_n(g, 18);
        for (uint32_t k = 0; k < n_body; k++) gen_body_op(g);
        // Statements end in stores to locals: spill values nothing used yet.
        while (g->n_pending > 1) {
            IrValue ops[2];
            ops[0] = g->pending[--g->n_pending];
            ops[1] = g->ptrs[rnd_n(g, g->n_allocas)];
            emit(g, OPK_STORE, ops, 2, NULL, 0, NULL, 0);
        }
        if (bi + 1 == n_blocks) {
            IrValue rv = pick_int(g);
            emit(g, OPK_RETURN, &rv, 1, NULL, 0, NULL, 0);
        } else {
            emit_branch(g, blocks, n_blocks, bi);
        }
    }
}

static IrOp build_module(IrCtx *c, uint64_t target_ops, uint64_t *n_ops) {
    Gen g;
    g.c = c;
    g.rng = 0x12345678u;
    g.i1 = ir_type_int(c, 1);
    g.i32 = ir_type_int(c, 32);
    g.i64 = ir_type_int(c, 64);
    g.ptr = ir_type_ptr(c);
    g.n_ops = 0;
    g.n_ints = g.n_pending = g.n_bools = g.n_allocas = g.n_ptrs = 0;
    IrRegion region = ir_region_create(c);
    IrBlock body = ir_block_create(c);
    ir_region_append_block(c, region, body);
    IrOpSpec ms;
    ms.op_type = OPK_MODULE;
    ms.operands = NULL;
    ms.n_operands = 0;
    ms.result_types = NULL;
    ms.n_results = 0;
    ms.attrs = NULL;
    ms.n_attrs = 0;
    ms.regions = &region;
    ms.n_regions = 1;
    ms.succs = NULL;
    ms.n_succs = 0;
    ms.n_succ_operands = NULL;
    IrOp module = ir_op_create(c, &ms);
    uint32_t f = 0;
    while (g.n_ops < target_ops) gen_func(&g, body, f++);
    *n_ops = g.n_ops + 1;
    return module;
}

// --- walks ---

static inline uint64_t mix(uint64_t h, uint64_t x) {
    h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

static uint64_t walk_value(IrCtx *c, uint64_t h, IrValue v) {
    IrOp d = ir_value_def_op(c, v);
    h = mix(h, d ? ir_op_type(c, d) : 99);
    // A constant operand's value tells operands apart beyond their kind.
    if (d && ir_op_type(c, d) == OPK_CONST) h = mix(h, (uint64_t)ir_attr_int_value(c, ir_op_attr(c, d, 0)));
    return mix(h, ir_type_width(c, ir_value_type(c, v)));
}

static uint64_t walk_ops(IrCtx *c, IrOp module) {
    uint64_t h = 0;
    IrBlock body = ir_region_block(c, ir_op_region(c, module, 0), 0);
    uint32_t nf = ir_block_num_ops(c, body);
    for (uint32_t f = 0; f < nf; f++) {
        IrRegion r = ir_op_region(c, ir_block_op(c, body, f), 0);
        uint32_t nb = ir_region_num_blocks(c, r);
        for (uint32_t bi = 0; bi < nb; bi++) {
            IrBlock b = ir_region_block(c, r, bi);
            uint32_t no = ir_block_num_ops(c, b);
            for (uint32_t i = 0; i < no; i++) {
                IrOp op = ir_block_op(c, b, i);
                uint32_t n = ir_op_num_operands(c, op);
                h = mix(h, ir_op_type(c, op) * 16 + n);
                for (uint32_t k = 0; k < n; k++) h = walk_value(c, h, ir_op_operand(c, op, k));
                uint32_t ns = ir_op_num_succs(c, op);
                for (uint32_t s = 0; s < ns; s++) {
                    uint32_t nso = ir_op_num_succ_operands(c, op, s);
                    for (uint32_t k = 0; k < nso; k++)
                        h = walk_value(c, h, ir_op_succ_operand(c, op, s, k));
                }
            }
        }
    }
    return h;
}

// The uses of one value are summed (use-list order differs between designs),
// and the per-value sums are mixed in value order.
static uint64_t walk_uses_of(IrCtx *c, uint64_t h, IrValue v) {
    uint64_t sum = 0;
    for (IrUse u = ir_value_first_use(c, v); u; u = ir_use_next(c, u))
        sum += ir_op_type(c, ir_use_owner(c, u)) * 1000003ull + ir_use_operand_index(c, u) + 1;
    return mix(h, sum);
}

static uint64_t walk_uses(IrCtx *c, IrOp module) {
    uint64_t sum = 0;
    IrBlock body = ir_region_block(c, ir_op_region(c, module, 0), 0);
    uint32_t nf = ir_block_num_ops(c, body);
    for (uint32_t f = 0; f < nf; f++) {
        IrRegion r = ir_op_region(c, ir_block_op(c, body, f), 0);
        uint32_t nb = ir_region_num_blocks(c, r);
        for (uint32_t bi = 0; bi < nb; bi++) {
            IrBlock b = ir_region_block(c, r, bi);
            uint32_t na = ir_block_num_args(c, b);
            for (uint32_t k = 0; k < na; k++) sum = walk_uses_of(c, sum, ir_block_arg(c, b, k));
            uint32_t no = ir_block_num_ops(c, b);
            for (uint32_t i = 0; i < no; i++) {
                IrOp op = ir_block_op(c, b, i);
                uint32_t nr = ir_op_num_results(c, op);
                for (uint32_t k = 0; k < nr; k++) sum = walk_uses_of(c, sum, ir_op_result(c, op, k));
            }
        }
    }
    return sum;
}

// --- rewrite: constant folding, then DCE ---

static int is_const(IrCtx *c, IrValue v, int64_t *out) {
    IrOp d = ir_value_def_op(c, v);
    if (!d || ir_op_type(c, d) != OPK_CONST) return 0;
    *out = ir_attr_int_value(c, ir_op_attr(c, d, 0));
    return 1;
}

static int is_pure(uint32_t t) {
    return t == OPK_CONST || t == OPK_GEP || t == OPK_LOAD || t == OPK_ICMP || t == OPK_ADD ||
           t == OPK_SUB || t == OPK_MUL || t == OPK_SEXT || t == OPK_SELECT || t == OPK_ALLOCA;
}

static uint32_t fold_block(IrCtx *c, IrBlock b, IrType i1) {
    uint32_t folded = 0;
    for (uint32_t i = 0; i < ir_block_num_ops(c, b); i++) {
        IrOp op = ir_block_op(c, b, i);
        uint32_t t = ir_op_type(c, op);
        if (t != OPK_ADD && t != OPK_SUB && t != OPK_MUL && t != OPK_ICMP) continue;
        int64_t x, y;
        if (!is_const(c, ir_op_operand(c, op, 0), &x) || !is_const(c, ir_op_operand(c, op, 1), &y))
            continue;
        uint32_t ux = (uint32_t)x, uy = (uint32_t)y;
        int64_t r;
        IrType rt = ir_value_type(c, ir_op_result(c, op, 0));
        if (t == OPK_ADD) r = (int32_t)(ux + uy);
        else if (t == OPK_SUB) r = (int32_t)(ux - uy);
        else if (t == OPK_MUL) r = (int32_t)(ux * uy);
        else {
            int64_t p = ir_attr_int_value(c, ir_op_attr(c, op, 0));
            int32_t sx = (int32_t)ux, sy = (int32_t)uy;
            r = p == 0 ? sx == sy : p == 1 ? sx != sy : p == 2 ? sx < sy : sx > sy;
            rt = i1;
        }
        IrAttr a = ir_attr_int(c, ATTR_VALUE, r, rt);
        IrOpSpec s;
        s.op_type = OPK_CONST;
        s.operands = NULL;
        s.n_operands = 0;
        s.result_types = &rt;
        s.n_results = 1;
        s.attrs = &a;
        s.n_attrs = 1;
        s.regions = NULL;
        s.n_regions = 0;
        s.succs = NULL;
        s.n_succs = 0;
        s.n_succ_operands = NULL;
        IrOp k = ir_op_create(c, &s);
        ir_block_insert_op(c, b, i, k);   // k at i, op at i + 1
        ir_replace_all_uses(c, ir_op_result(c, op, 0), ir_op_result(c, k, 0));
        ir_op_erase(c, op);               // the next op is now at i + 1
        folded++;
    }
    return folded;
}

static uint32_t dce_region(IrCtx *c, IrRegion r) {
    uint32_t erased = 0;
    uint32_t changed = 1;
    while (changed) {
        changed = 0;
        uint32_t nb = ir_region_num_blocks(c, r);
        for (uint32_t bi = nb; bi-- > 0;) {
            IrBlock b = ir_region_block(c, r, bi);
            for (uint32_t i = ir_block_num_ops(c, b); i-- > 0;) {
                IrOp op = ir_block_op(c, b, i);
                if (!is_pure(ir_op_type(c, op))) continue;
                uint32_t nr = ir_op_num_results(c, op);
                uint32_t used = 0;
                for (uint32_t k = 0; k < nr; k++) used |= ir_value_num_uses(c, ir_op_result(c, op, k));
                if (used) continue;
                ir_op_erase(c, op);
                erased++;
                changed = 1;
            }
        }
    }
    return erased;
}

static void rewrite(IrCtx *c, IrOp module, uint32_t *folded, uint32_t *erased) {
    IrType i1 = ir_type_int(c, 1);
    *folded = 0;
    *erased = 0;
    IrBlock body = ir_region_block(c, ir_op_region(c, module, 0), 0);
    uint32_t nf = ir_block_num_ops(c, body);
    for (uint32_t f = 0; f < nf; f++) {
        IrRegion r = ir_op_region(c, ir_block_op(c, body, f), 0);
        uint32_t nb = ir_region_num_blocks(c, r);
        for (uint32_t bi = 0; bi < nb; bi++) *folded += fold_block(c, ir_region_block(c, r, bi), i1);
        *erased += dce_region(c, r);
    }
}

int main(int argc, char **argv) {
#ifdef BENCH_HOSTED
    platform_init(argc, argv, NULL);
#else
    // tinyc's ELF crt0 has already called platform_init but does not pass
    // argc/argv on to main.
    (void)argc;
    (void)argv;
#endif
    size_t nargs, args_size;
    platform_args_sizes_get(&nargs, &args_size);
    char *args_buf = (char *)malloc(args_size + 1);
    char **args = (char **)malloc((nargs + 1) * sizeof(char *));
    platform_args_get(args, args_buf);
    uint64_t target = nargs > 1 ? (uint64_t)atoi(args[1]) : 100000;
    int reps = nargs > 2 ? atoi(args[2]) : 3;
    int reserve = nargs > 3 ? atoi(args[3]) : 0;
    for (int rep = 0; rep <= reps; rep++) {
        IrCtx *c = ir_ctx_create();
        if (reserve) ir_ctx_reserve(c, (uint32_t)(target + target / 8));
        uint64_t n_ops;
        uint32_t folded, erased;
        uint64_t t0 = now_ns();
        IrOp module = build_module(c, target, &n_ops);
        uint64_t t1 = now_ns();
        uint64_t h_walk = walk_ops(c, module);
        uint64_t t2 = now_ns();
        uint64_t h_uses = walk_uses(c, module);
        uint64_t t3 = now_ns();
        rewrite(c, module, &folded, &erased);
        uint64_t t4 = now_ns();
        uint64_t h_walk2 = walk_ops(c, module);
        uint64_t h_uses2 = walk_uses(c, module);
        uint64_t bytes = ir_ctx_bytes(c);
        ir_ctx_destroy(c);
        printf("impl=%s rep=%d ops=%llu build_ns=%llu walk_ns=%llu uses_ns=%llu rewrite_ns=%llu "
               "bytes=%llu folded=%u erased=%u h=%llx,%llx,%llx,%llx\n",
               ir_impl_name(), rep, (unsigned long long)n_ops,
               (unsigned long long)(t1 - t0), (unsigned long long)(t2 - t1),
               (unsigned long long)(t3 - t2), (unsigned long long)(t4 - t3),
               (unsigned long long)bytes, folded, erased,
               (unsigned long long)h_walk, (unsigned long long)h_uses,
               (unsigned long long)h_walk2, (unsigned long long)h_uses2);
    }
    return 0;
}

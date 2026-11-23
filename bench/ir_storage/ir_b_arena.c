// Design B, per-op arena arrays: 32-bit index handles for ops, values,
// blocks, regions, types and attributes (context-owned vectors that double by
// reallocation, index 0 reserved), but the variable-size parts of an op live
// in one arena array of 32-bit words that the op points to:
//   [operand value indices (regular, then successor operands)]
//   [attribute handles][region handles][(block, operand count) per successor]
// An operand is 4 bytes and has no links. Instead each value keeps an arena
// array of its uses, (owner op, flat operand index) pairs, that doubles when
// full; removing a use is a search plus swap-with-last. A use handle packs
// (value, position + 1). An op's results are consecutive values.
#include "ir_api.h"
#include <base/arena.h>

#if IR_HANDLE_BITS != 32
#error "the B designs use 32-bit index handles: build with IR_HANDLE_BITS=32"
#endif
#include <base/buddy.h>
#include <string.h>

typedef struct BType { uint32_t kind; uint32_t width; } BType;
typedef struct BAttr { uint32_t kind; uint32_t key; uint32_t type; uint32_t pad; int64_t value; } BAttr;
typedef struct BUse { uint32_t owner; uint32_t slot; } BUse;

#define VAL_BLOCK_ARG 0x80000000u

typedef struct BValue {
    uint32_t type;
    uint32_t def;
    uint32_t kind_index;
    uint32_t n_uses;
    uint32_t cap_uses;
    uint32_t location;
    BUse *uses;
} BValue;

typedef struct BOp {
    uint16_t op_type;
    uint16_t n_attrs;
    uint16_t n_regions;
    uint16_t n_succs;
    uint32_t n_operands;
    uint32_t n_succ_operands;
    uint32_t n_results;
    uint32_t first_result;
    uint32_t parent;
    uint32_t location;
    uint32_t *tail;
} BOp;

typedef struct BBlock {
    uint32_t *ops;
    uint32_t n_ops, cap_ops;
    uint32_t *args;
    uint32_t n_args, cap_args;
    uint32_t parent;
    uint32_t pad;
} BBlock;

typedef struct BRegion {
    uint32_t *blocks;
    uint32_t n_blocks, cap_blocks;
    uint32_t parent;
    uint32_t pad;
} BRegion;

#define DECL_VEC(T, name) T *name; uint32_t n_##name, cap_##name
#define EL(c, name, i) ((c)->name[i])

struct IrCtx {
    DECL_VEC(BOp, ops);
    DECL_VEC(BValue, vals);
    DECL_VEC(BBlock, blocks);
    DECL_VEC(BRegion, regions);
    DECL_VEC(BType, types);
    DECL_VEC(BAttr, attrs);
    Arena *arena;
    uint64_t arena_bytes;
    uint32_t location;
};

const char *ir_impl_name(void) { return "B-arena"; }

static void *vec_grow(void *old, uint32_t n_used, uint32_t *cap, uint32_t elem, uint32_t need) {
    size_t want = *cap ? (size_t)*cap * 2 : 64;
    while (want < need) want *= 2;
    size_t actual;
    void *p = buddy_alloc(want * elem, &actual);
    if (n_used) memcpy(p, old, (size_t)n_used * elem);
    if (old) buddy_free(old);
    *cap = (uint32_t)(actual / elem);
    return p;
}

#define VEC_RESERVE(c, name, extra_n) do { \
    if ((c)->n_##name + (extra_n) > (c)->cap_##name) \
        (c)->name = vec_grow((c)->name, (c)->n_##name, &(c)->cap_##name, \
                             (uint32_t)sizeof(*(c)->name), (c)->n_##name + (extra_n)); \
} while (0)

#define VEC_INIT(c, name) do { \
    (c)->name = NULL; (c)->n_##name = 0; (c)->cap_##name = 0; \
    VEC_RESERVE(c, name, 1); \
    memset(&(c)->name[0], 0, sizeof(*(c)->name)); \
    (c)->n_##name = 1; \
} while (0)

#define VEC_FREE(c, name) do { if ((c)->name) buddy_free((c)->name); } while (0)
// Bytes in use, like the arena designs' count; the doubling slack shows up
// in peak RSS instead.
#define VEC_BYTES(c, name) ((uint64_t)(c)->n_##name * sizeof(*(c)->name))

static void *a_alloc(IrCtx *c, size_t n) {
    c->arena_bytes += (n + 15) & ~(size_t)15;
    return arena_alloc(c->arena, n);
}

IrCtx *ir_ctx_create(void) {
    Arena *arena = arena_create(64 * 1024 * 1024 - 64 * 1024);
    IrCtx *c = (IrCtx *)arena_alloc(arena, sizeof(IrCtx));
    memset(c, 0, sizeof(IrCtx));
    c->arena = arena;
    VEC_INIT(c, ops);
    VEC_INIT(c, vals);
    VEC_INIT(c, blocks);
    VEC_INIT(c, regions);
    VEC_INIT(c, types);
    VEC_INIT(c, attrs);
    c->location = 1;
    return c;
}

void ir_ctx_reserve(IrCtx *c, uint32_t n_ops) {
    VEC_RESERVE(c, ops, n_ops);
    VEC_RESERVE(c, vals, n_ops);
    VEC_RESERVE(c, attrs, n_ops / 2);
    VEC_RESERVE(c, blocks, n_ops / 8);
    VEC_RESERVE(c, regions, n_ops / 64);
}

void ir_ctx_destroy(IrCtx *c) {
    VEC_FREE(c, ops);
    VEC_FREE(c, vals);
    VEC_FREE(c, blocks);
    VEC_FREE(c, regions);
    VEC_FREE(c, types);
    VEC_FREE(c, attrs);
    arena_destroy(c->arena);
}

uint64_t ir_ctx_bytes(IrCtx *c) {
    return c->arena_bytes + VEC_BYTES(c, ops) + VEC_BYTES(c, vals) + VEC_BYTES(c, blocks) +
           VEC_BYTES(c, regions) + VEC_BYTES(c, types) + VEC_BYTES(c, attrs);
}

static IrType intern(IrCtx *c, uint32_t kind, uint32_t width) {
    for (uint32_t i = 1; i < c->n_types; i++)
        if (c->types[i].kind == kind && c->types[i].width == width) return i;
    VEC_RESERVE(c, types, 1);
    uint32_t t = c->n_types++;
    c->types[t].kind = kind;
    c->types[t].width = width;
    return t;
}

IrType ir_type_int(IrCtx *c, uint32_t width) { return intern(c, 0, width); }
IrType ir_type_ptr(IrCtx *c) { return intern(c, 1, 0); }
uint32_t ir_type_width(IrCtx *c, IrType t) { return c->types[t].width; }

IrAttr ir_attr_int(IrCtx *c, uint32_t key, int64_t value, IrType type) {
    VEC_RESERVE(c, attrs, 1);
    uint32_t a = c->n_attrs++;
    BAttr *p = &c->attrs[a];
    p->kind = 0;
    p->key = key;
    p->type = type;
    p->pad = 0;
    p->value = value;
    return a;
}

int64_t ir_attr_int_value(IrCtx *c, IrAttr a) { return c->attrs[a].value; }

static uint32_t *u32_grow(IrCtx *c, uint32_t *old, uint32_t n, uint32_t *cap) {
    uint32_t nc = *cap ? *cap * 2 : 4;
    uint32_t *p = (uint32_t *)a_alloc(c, nc * sizeof(uint32_t));
    if (n) memcpy(p, old, n * sizeof(uint32_t));
    *cap = nc;
    return p;
}

IrRegion ir_region_create(IrCtx *c) {
    VEC_RESERVE(c, regions, 1);
    uint32_t r = c->n_regions++;
    BRegion *rp = &EL(c, regions, r);
    rp->blocks = NULL;
    rp->n_blocks = rp->cap_blocks = 0;
    rp->parent = 0;
    rp->pad = 0;
    return r;
}

void ir_region_append_block(IrCtx *c, IrRegion rh, IrBlock bh) {
    BRegion *r = &c->regions[rh];
    if (r->n_blocks == r->cap_blocks) r->blocks = u32_grow(c, r->blocks, r->n_blocks, &r->cap_blocks);
    r->blocks[r->n_blocks++] = bh;
    c->blocks[bh].parent = rh;
}

uint32_t ir_region_num_blocks(IrCtx *c, IrRegion r) { return c->regions[r].n_blocks; }
IrBlock ir_region_block(IrCtx *c, IrRegion r, uint32_t i) { return c->regions[r].blocks[i]; }

IrBlock ir_block_create(IrCtx *c) {
    VEC_RESERVE(c, blocks, 1);
    uint32_t b = c->n_blocks++;
    BBlock *bp = &EL(c, blocks, b);
    bp->ops = NULL;
    bp->n_ops = bp->cap_ops = 0;
    bp->args = NULL;
    bp->n_args = bp->cap_args = 0;
    bp->parent = 0;
    bp->pad = 0;
    return b;
}

static void init_value(BValue *p, uint32_t type, uint32_t def, uint32_t kind_index, uint32_t loc) {
    p->type = type;
    p->def = def;
    p->kind_index = kind_index;
    p->n_uses = 0;
    p->cap_uses = 0;
    p->location = loc;
    p->uses = NULL;
}

IrValue ir_block_add_arg(IrCtx *c, IrBlock bh, IrType t) {
    VEC_RESERVE(c, vals, 1);
    uint32_t v = c->n_vals++;
    BBlock *b = &c->blocks[bh];
    init_value(&c->vals[v], t, bh, VAL_BLOCK_ARG | b->n_args, c->location);
    if (b->n_args == b->cap_args) b->args = u32_grow(c, b->args, b->n_args, &b->cap_args);
    b->args[b->n_args++] = v;
    return v;
}

uint32_t ir_block_num_args(IrCtx *c, IrBlock b) { return c->blocks[b].n_args; }
IrValue ir_block_arg(IrCtx *c, IrBlock b, uint32_t i) { return c->blocks[b].args[i]; }
uint32_t ir_block_num_ops(IrCtx *c, IrBlock b) { return c->blocks[b].n_ops; }
IrOp ir_block_op(IrCtx *c, IrBlock b, uint32_t i) { return c->blocks[b].ops[i]; }

void ir_block_append_op(IrCtx *c, IrBlock bh, IrOp op) {
    BBlock *b = &c->blocks[bh];
    if (b->n_ops == b->cap_ops) b->ops = u32_grow(c, b->ops, b->n_ops, &b->cap_ops);
    b->ops[b->n_ops++] = op;
    c->ops[op].parent = bh;
}

void ir_block_insert_op(IrCtx *c, IrBlock bh, uint32_t idx, IrOp op) {
    BBlock *b = &c->blocks[bh];
    if (b->n_ops == b->cap_ops) b->ops = u32_grow(c, b->ops, b->n_ops, &b->cap_ops);
    if (idx < b->n_ops)
        memmove(b->ops + idx + 1, b->ops + idx, (b->n_ops - idx) * sizeof(uint32_t));
    b->ops[idx] = op;
    b->n_ops++;
    c->ops[op].parent = bh;
}

static void use_add(IrCtx *c, uint32_t v, uint32_t owner, uint32_t slot) {
    BValue *val = &c->vals[v];
    if (val->n_uses == val->cap_uses) {
        uint32_t nc = val->cap_uses ? val->cap_uses * 2 : 2;
        BUse *nu = (BUse *)a_alloc(c, nc * sizeof(BUse));
        if (val->n_uses) memcpy(nu, val->uses, val->n_uses * sizeof(BUse));
        val->uses = nu;
        val->cap_uses = nc;
    }
    val->uses[val->n_uses].owner = owner;
    val->uses[val->n_uses].slot = slot;
    val->n_uses++;
}

static void use_remove(IrCtx *c, uint32_t v, uint32_t owner, uint32_t slot) {
    BValue *val = &c->vals[v];
    BUse *u = val->uses;
    uint32_t n = val->n_uses;
    for (uint32_t i = 0; i < n; i++) {
        if (u[i].owner == owner && u[i].slot == slot) {
            u[i] = u[n - 1];
            val->n_uses = n - 1;
            return;
        }
    }
}

IrOp ir_op_create(IrCtx *c, const IrOpSpec *s) {
    uint32_t n_succ_ops = 0;
    for (uint32_t k = 0; k < s->n_succs; k++) n_succ_ops += s->n_succ_operands[k];
    uint32_t n_all = s->n_operands + n_succ_ops;
    uint32_t n_words = n_all + s->n_attrs + s->n_regions + 2 * s->n_succs;
    VEC_RESERVE(c, ops, 1);
    VEC_RESERVE(c, vals, s->n_results);
    uint32_t op = c->n_ops++;
    BOp *o = &c->ops[op];
    o->op_type = (uint16_t)s->op_type;
    o->n_attrs = (uint16_t)s->n_attrs;
    o->n_regions = (uint16_t)s->n_regions;
    o->n_succs = (uint16_t)s->n_succs;
    o->n_operands = s->n_operands;
    o->n_succ_operands = n_succ_ops;
    o->n_results = s->n_results;
    o->first_result = c->n_vals;
    o->parent = 0;
    o->location = c->location;
    uint32_t *t = n_words ? (uint32_t *)a_alloc(c, n_words * sizeof(uint32_t)) : NULL;
    o->tail = t;
    for (uint32_t i = 0; i < n_all; i++) {
        t[i] = s->operands[i];
        use_add(c, s->operands[i], op, i);
    }
    for (uint32_t i = 0; i < s->n_results; i++)
        init_value(&c->vals[c->n_vals + i], s->result_types[i], op, i, c->location);
    c->n_vals += s->n_results;
    uint32_t *ex = t + n_all;
    for (uint32_t i = 0; i < s->n_attrs; i++) *ex++ = s->attrs[i];
    for (uint32_t i = 0; i < s->n_regions; i++) {
        *ex++ = s->regions[i];
        c->regions[s->regions[i]].parent = op;
    }
    for (uint32_t i = 0; i < s->n_succs; i++) {
        *ex++ = s->succs[i];
        *ex++ = s->n_succ_operands[i];
    }
    return op;
}

static inline uint32_t *op_extra(BOp *o) { return o->tail + o->n_operands + o->n_succ_operands; }

uint32_t ir_op_type(IrCtx *c, IrOp o) { return c->ops[o].op_type; }
uint32_t ir_op_num_operands(IrCtx *c, IrOp o) { return c->ops[o].n_operands; }
IrValue ir_op_operand(IrCtx *c, IrOp o, uint32_t i) { return c->ops[o].tail[i]; }
uint32_t ir_op_num_results(IrCtx *c, IrOp o) { return c->ops[o].n_results; }
IrValue ir_op_result(IrCtx *c, IrOp o, uint32_t i) { return c->ops[o].first_result + i; }
uint32_t ir_op_num_attrs(IrCtx *c, IrOp o) { return c->ops[o].n_attrs; }
IrAttr ir_op_attr(IrCtx *c, IrOp o, uint32_t i) { return op_extra(&c->ops[o])[i]; }
uint32_t ir_op_num_regions(IrCtx *c, IrOp o) { return c->ops[o].n_regions; }

IrRegion ir_op_region(IrCtx *c, IrOp oh, uint32_t i) {
    BOp *o = &c->ops[oh];
    return op_extra(o)[o->n_attrs + i];
}

uint32_t ir_op_num_succs(IrCtx *c, IrOp o) { return c->ops[o].n_succs; }

IrBlock ir_op_succ(IrCtx *c, IrOp oh, uint32_t s) {
    BOp *o = &c->ops[oh];
    return op_extra(o)[o->n_attrs + o->n_regions + 2 * s];
}

uint32_t ir_op_num_succ_operands(IrCtx *c, IrOp oh, uint32_t s) {
    BOp *o = &c->ops[oh];
    return op_extra(o)[o->n_attrs + o->n_regions + 2 * s + 1];
}

IrValue ir_op_succ_operand(IrCtx *c, IrOp oh, uint32_t s, uint32_t i) {
    BOp *o = &c->ops[oh];
    const uint32_t *su = op_extra(o) + o->n_attrs + o->n_regions;
    uint32_t off = o->n_operands;
    for (uint32_t k = 0; k < s; k++) off += su[2 * k + 1];
    return o->tail[off + i];
}

IrBlock ir_op_parent_block(IrCtx *c, IrOp o) { return c->ops[o].parent; }

IrType ir_value_type(IrCtx *c, IrValue v) { return c->vals[v].type; }

IrOp ir_value_def_op(IrCtx *c, IrValue vh) {
    BValue *v = &c->vals[vh];
    return (v->kind_index & VAL_BLOCK_ARG) ? 0 : v->def;
}

uint32_t ir_value_num_uses(IrCtx *c, IrValue v) { return c->vals[v].n_uses; }

IrUse ir_value_first_use(IrCtx *c, IrValue v) {
    return c->vals[v].n_uses ? (((IrUse)v << 32) | 1) : 0;
}

IrUse ir_use_next(IrCtx *c, IrUse u) {
    uint32_t v = (uint32_t)(u >> 32);
    uint32_t pos = (uint32_t)u;   // position + 1
    return pos < c->vals[v].n_uses ? u + 1 : 0;
}

IrOp ir_use_owner(IrCtx *c, IrUse u) {
    return c->vals[(uint32_t)(u >> 32)].uses[(uint32_t)u - 1].owner;
}

uint32_t ir_use_operand_index(IrCtx *c, IrUse u) {
    return c->vals[(uint32_t)(u >> 32)].uses[(uint32_t)u - 1].slot;
}

void ir_op_set_operand(IrCtx *c, IrOp op, uint32_t idx, IrValue v) {
    uint32_t *t = c->ops[op].tail;
    uint32_t old = t[idx];
    if (old == v) return;
    use_remove(c, old, op, idx);
    t[idx] = v;
    use_add(c, v, op, idx);
}

void ir_replace_all_uses(IrCtx *c, IrValue from, IrValue to) {
    if (from == to) return;
    uint32_t n = c->vals[from].n_uses;
    for (uint32_t i = 0; i < n; i++) {
        // use_add may grow vals[to].uses but never moves vals[from].
        BUse u = c->vals[from].uses[i];
        c->ops[u.owner].tail[u.slot] = to;
        use_add(c, to, u.owner, u.slot);
    }
    c->vals[from].n_uses = 0;
}

void ir_op_erase(IrCtx *c, IrOp op) {
    BOp *o = &c->ops[op];
    uint32_t n_all = o->n_operands + o->n_succ_operands;
    for (uint32_t i = 0; i < n_all; i++) {
        if (o->tail[i]) use_remove(c, o->tail[i], op, i);
        o->tail[i] = 0;
    }
    if (o->parent) {
        BBlock *b = &c->blocks[o->parent];
        uint32_t w = 0;
        for (uint32_t i = 0; i < b->n_ops; i++)
            if (b->ops[i] != op) b->ops[w++] = b->ops[i];
        b->n_ops = w;
    }
    o->parent = 0;
}

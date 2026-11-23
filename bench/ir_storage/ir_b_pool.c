// Design B, shared pools: 32-bit index handles into context-owned vectors.
//
// The context owns one vector per kind (ops, values, blocks, regions, types,
// attributes) plus two shared pools for the variable-size parts:
//   - operands: {value, owner, next_use, prev_use}, 16 bytes, all 32-bit
//     indices; an op's operands (regular, then successor operands) are
//     consecutive, and the op stores the start. Use lists are threaded through
//     operand indices, so a use handle is an operand index.
//   - extras: 32-bit words; an op's attribute handles, region handles and
//     (block, operand count) successor pairs are consecutive.
// An op's results are consecutive entries of the value vector, so
// ir_op_result is first_result + i without touching the value.
//
// Each vector doubles by reallocation (buddy_alloc, copy, buddy_free), so an
// index stays valid and a pointer into a vector does not. Index 0 of every
// vector is reserved as the invalid handle. Erased ops stay as empty slots.
// Built with -DB_SEGMENTED (B-pool-seg), each vector is instead a table of
// fixed-size arena segments, so it grows without copying.
// A block's op list and argument list are arena arrays of 32-bit indices.
#include "ir_api.h"
#include <base/arena.h>

#if IR_HANDLE_BITS != 32
#error "the B designs use 32-bit index handles: build with IR_HANDLE_BITS=32"
#endif
#include <base/buddy.h>
#include <string.h>

typedef struct BType { uint32_t kind; uint32_t width; } BType;
typedef struct BAttr { uint32_t kind; uint32_t key; uint32_t type; uint32_t pad; int64_t value; } BAttr;

typedef struct BOperand { uint32_t value, owner, next, prev; } BOperand;

#define VAL_BLOCK_ARG 0x80000000u

typedef struct BValue {
    uint32_t type;
    uint32_t def;          // op index for a result, block index for a block arg
    uint32_t kind_index;   // VAL_BLOCK_ARG | index, or the result index
    uint32_t n_uses;
    uint32_t first_use;    // operand index
    uint32_t location;
} BValue;

typedef struct BOp {
    uint16_t op_type;
    uint16_t n_attrs;
    uint16_t n_regions;
    uint16_t n_succs;
    uint32_t n_operands;       // regular operands
    uint32_t n_succ_operands;  // all successor operands
    uint32_t n_results;
    uint32_t first_operand;
    uint32_t first_result;
    uint32_t first_extra;
    uint32_t parent;           // block
    uint32_t location;
} BOp;

typedef struct BBlock {
    uint32_t *ops;
    uint32_t n_ops, cap_ops;
    uint32_t *args;
    uint32_t n_args, cap_args;
    uint32_t parent;           // region
    uint32_t pad;
} BBlock;

typedef struct BRegion {
    uint32_t *blocks;
    uint32_t n_blocks, cap_blocks;
    uint32_t parent;           // op
    uint32_t pad;
} BRegion;

#ifdef B_SEGMENTED
// Segmented vectors: fixed-size segments of SEG_N elements carved from the
// arena plus a small table of segment pointers. Nothing is ever copied or
// moved; an op's operand/result/extra range never straddles two segments.
#define SEG_SHIFT 12
#define SEG_N (1u << SEG_SHIFT)
#define SEG_MASK (SEG_N - 1)
#define DECL_VEC(T, name) T **name##_seg; uint32_t n_##name, cap_##name, nseg_##name, segcap_##name
#define EL(c, name, i) ((c)->name##_seg[(i) >> SEG_SHIFT][(i) & SEG_MASK])
#else
#define DECL_VEC(T, name) T *name; uint32_t n_##name, cap_##name
#define EL(c, name, i) ((c)->name[i])
#endif

struct IrCtx {
    DECL_VEC(BOp, ops);
    DECL_VEC(BValue, vals);
    DECL_VEC(BOperand, opnds);
    DECL_VEC(uint32_t, extra);
    DECL_VEC(BBlock, blocks);
    DECL_VEC(BRegion, regions);
    DECL_VEC(BType, types);
    DECL_VEC(BAttr, attrs);
    Arena *arena;
    uint64_t arena_bytes;
    uint32_t location;
};

#ifdef B_SEGMENTED
const char *ir_impl_name(void) { return "B-pool-seg"; }
#else
const char *ir_impl_name(void) { return "B-pool"; }
#endif

static void *a_alloc(IrCtx *c, size_t n);

#ifdef B_SEGMENTED
// Appends one segment; returns the (possibly new) segment table. The table
// travels as void * because tinyc has no `void ***`.
static void *seg_add(IrCtx *c, void *table_, uint32_t *nseg, uint32_t *segcap, uint32_t *cap,
                     size_t elem) {
    void **table = (void **)table_;
    if (*nseg == *segcap) {
        uint32_t nc = *segcap ? *segcap * 2 : 16;
        void **nt = (void **)arena_alloc(c->arena, nc * sizeof(void *));
        if (*nseg) memcpy(nt, table, *nseg * sizeof(void *));
        table = nt;
        *segcap = nc;
    }
    // Not counted in arena_bytes: ir_ctx_bytes counts the elements in use.
    table[(*nseg)++] = arena_alloc(c->arena, SEG_N * elem);
    *cap += SEG_N;
    return table;
}

#define VEC_RESERVE(c, name, extra_n) do { \
    uint32_t off_ = (c)->n_##name & SEG_MASK; \
    if (off_ && off_ + (extra_n) > SEG_N) (c)->n_##name += SEG_N - off_; \
    while ((c)->n_##name + (extra_n) > (c)->cap_##name) \
        (c)->name##_seg = seg_add((c), (void *)(c)->name##_seg, &(c)->nseg_##name, \
                                  &(c)->segcap_##name, &(c)->cap_##name, \
                                  sizeof(**(c)->name##_seg)); \
} while (0)

#define VEC_INIT(c, name) do { \
    (c)->name##_seg = NULL; (c)->n_##name = 0; (c)->cap_##name = 0; \
    (c)->nseg_##name = 0; (c)->segcap_##name = 0; \
    VEC_RESERVE(c, name, 1); \
    memset(&EL(c, name, 0), 0, sizeof(**(c)->name##_seg)); \
    (c)->n_##name = 1; \
} while (0)

#define VEC_FREE(c, name) do { } while (0)
#define VEC_BYTES(c, name) ((uint64_t)(c)->n_##name * sizeof(**(c)->name##_seg))
#else
// Grow a vector so it holds at least `need` elements. Capacity doubles and
// uses all of the buddy block it gets.
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
#endif

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
    VEC_INIT(c, opnds);
    VEC_INIT(c, extra);
    VEC_INIT(c, blocks);
    VEC_INIT(c, regions);
    VEC_INIT(c, types);
    VEC_INIT(c, attrs);
    c->location = 1;
    return c;
}

void ir_ctx_reserve(IrCtx *c, uint32_t n_ops) {
#ifdef B_SEGMENTED
    (void)c;
    (void)n_ops;
    return;
#endif
    VEC_RESERVE(c, ops, n_ops);
    VEC_RESERVE(c, vals, n_ops);
    VEC_RESERVE(c, attrs, n_ops / 2);
    VEC_RESERVE(c, opnds, n_ops * 2);
    VEC_RESERVE(c, extra, n_ops);
    VEC_RESERVE(c, blocks, n_ops / 8);
    VEC_RESERVE(c, regions, n_ops / 64);
}

void ir_ctx_destroy(IrCtx *c) {
    VEC_FREE(c, ops);
    VEC_FREE(c, vals);
    VEC_FREE(c, opnds);
    VEC_FREE(c, extra);
    VEC_FREE(c, blocks);
    VEC_FREE(c, regions);
    VEC_FREE(c, types);
    VEC_FREE(c, attrs);
    arena_destroy(c->arena);
}

uint64_t ir_ctx_bytes(IrCtx *c) {
    return c->arena_bytes + VEC_BYTES(c, ops) + VEC_BYTES(c, vals) + VEC_BYTES(c, opnds) +
           VEC_BYTES(c, extra) + VEC_BYTES(c, blocks) + VEC_BYTES(c, regions) +
           VEC_BYTES(c, types) + VEC_BYTES(c, attrs);
}

static IrType intern(IrCtx *c, uint32_t kind, uint32_t width) {
    for (uint32_t i = 1; i < c->n_types; i++)
        if (EL(c, types, i).kind == kind && EL(c, types, i).width == width) return i;
    VEC_RESERVE(c, types, 1);
    uint32_t t = c->n_types++;
    EL(c, types, t).kind = kind;
    EL(c, types, t).width = width;
    return t;
}

IrType ir_type_int(IrCtx *c, uint32_t width) { return intern(c, 0, width); }
IrType ir_type_ptr(IrCtx *c) { return intern(c, 1, 0); }
uint32_t ir_type_width(IrCtx *c, IrType t) { return EL(c, types, t).width; }

IrAttr ir_attr_int(IrCtx *c, uint32_t key, int64_t value, IrType type) {
    VEC_RESERVE(c, attrs, 1);
    uint32_t a = c->n_attrs++;
    BAttr *p = &EL(c, attrs, a);
    p->kind = 0;
    p->key = key;
    p->type = type;
    p->pad = 0;
    p->value = value;
    return a;
}

int64_t ir_attr_int_value(IrCtx *c, IrAttr a) { return EL(c, attrs, a).value; }

// Arena array of 32-bit indices that doubles; the old array is left behind.
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
    BRegion *r = &EL(c, regions, rh);
    if (r->n_blocks == r->cap_blocks) r->blocks = u32_grow(c, r->blocks, r->n_blocks, &r->cap_blocks);
    r->blocks[r->n_blocks++] = bh;
    EL(c, blocks, bh).parent = rh;
}

uint32_t ir_region_num_blocks(IrCtx *c, IrRegion r) { return EL(c, regions, r).n_blocks; }
IrBlock ir_region_block(IrCtx *c, IrRegion r, uint32_t i) { return EL(c, regions, r).blocks[i]; }

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

IrValue ir_block_add_arg(IrCtx *c, IrBlock bh, IrType t) {
    VEC_RESERVE(c, vals, 1);
    uint32_t v = c->n_vals++;
    BBlock *b = &EL(c, blocks, bh);
    BValue *p = &EL(c, vals, v);
    p->type = t;
    p->def = bh;
    p->kind_index = VAL_BLOCK_ARG | b->n_args;
    p->n_uses = 0;
    p->first_use = 0;
    p->location = c->location;
    if (b->n_args == b->cap_args) b->args = u32_grow(c, b->args, b->n_args, &b->cap_args);
    b->args[b->n_args++] = v;
    return v;
}

uint32_t ir_block_num_args(IrCtx *c, IrBlock b) { return EL(c, blocks, b).n_args; }
IrValue ir_block_arg(IrCtx *c, IrBlock b, uint32_t i) { return EL(c, blocks, b).args[i]; }
uint32_t ir_block_num_ops(IrCtx *c, IrBlock b) { return EL(c, blocks, b).n_ops; }
IrOp ir_block_op(IrCtx *c, IrBlock b, uint32_t i) { return EL(c, blocks, b).ops[i]; }

void ir_block_append_op(IrCtx *c, IrBlock bh, IrOp op) {
    BBlock *b = &EL(c, blocks, bh);
    if (b->n_ops == b->cap_ops) b->ops = u32_grow(c, b->ops, b->n_ops, &b->cap_ops);
    b->ops[b->n_ops++] = op;
    EL(c, ops, op).parent = bh;
}

void ir_block_insert_op(IrCtx *c, IrBlock bh, uint32_t idx, IrOp op) {
    BBlock *b = &EL(c, blocks, bh);
    if (b->n_ops == b->cap_ops) b->ops = u32_grow(c, b->ops, b->n_ops, &b->cap_ops);
    if (idx < b->n_ops)
        memmove(b->ops + idx + 1, b->ops + idx, (b->n_ops - idx) * sizeof(uint32_t));
    b->ops[idx] = op;
    b->n_ops++;
    EL(c, ops, op).parent = bh;
}

static inline void use_push(IrCtx *c, uint32_t v, uint32_t u) {
    BOperand *o = &EL(c, opnds, u);
    BValue *val = &EL(c, vals, v);
    o->value = v;
    o->prev = 0;
    o->next = val->first_use;
    if (val->first_use) EL(c, opnds, val->first_use).prev = u;
    val->first_use = u;
    val->n_uses++;
}

static inline void use_unlink(IrCtx *c, uint32_t u) {
    BOperand *o = &EL(c, opnds, u);
    if (!o->value) return;
    if (o->prev) EL(c, opnds, o->prev).next = o->next;
    else         EL(c, vals, o->value).first_use = o->next;
    if (o->next) EL(c, opnds, o->next).prev = o->prev;
    EL(c, vals, o->value).n_uses--;
    o->value = 0;
    o->next = 0;
    o->prev = 0;
}

IrOp ir_op_create(IrCtx *c, const IrOpSpec *s) {
    uint32_t n_succ_ops = 0;
    for (uint32_t k = 0; k < s->n_succs; k++) n_succ_ops += s->n_succ_operands[k];
    uint32_t n_all = s->n_operands + n_succ_ops;
    uint32_t n_extra = s->n_attrs + s->n_regions + 2 * s->n_succs;
    // Reserve first: growing a vector moves it.
    VEC_RESERVE(c, ops, 1);
    VEC_RESERVE(c, vals, s->n_results);
    VEC_RESERVE(c, opnds, n_all);
    VEC_RESERVE(c, extra, n_extra);

    uint32_t op = c->n_ops++;
    BOp *o = &EL(c, ops, op);
    o->op_type = (uint16_t)s->op_type;
    o->n_attrs = (uint16_t)s->n_attrs;
    o->n_regions = (uint16_t)s->n_regions;
    o->n_succs = (uint16_t)s->n_succs;
    o->n_operands = s->n_operands;
    o->n_succ_operands = n_succ_ops;
    o->n_results = s->n_results;
    o->first_operand = c->n_opnds;
    o->first_result = c->n_vals;
    o->first_extra = c->n_extra;
    o->parent = 0;
    o->location = c->location;

    uint32_t first = c->n_opnds;
    c->n_opnds += n_all;
    for (uint32_t i = 0; i < n_all; i++) {
        EL(c, opnds, first + i).owner = op;
        use_push(c, s->operands[i], first + i);
    }
    // Take element addresses only when the range is non-empty: an empty
    // range may sit exactly at the end of the last segment.
    BValue *res = s->n_results ? &EL(c, vals, c->n_vals) : NULL;
    for (uint32_t i = 0; i < s->n_results; i++) {
        res[i].type = s->result_types[i];
        res[i].def = op;
        res[i].kind_index = i;
        res[i].n_uses = 0;
        res[i].first_use = 0;
        res[i].location = c->location;
    }
    c->n_vals += s->n_results;
    uint32_t *ex = n_extra ? &EL(c, extra, c->n_extra) : NULL;
    c->n_extra += n_extra;
    for (uint32_t i = 0; i < s->n_attrs; i++) *ex++ = s->attrs[i];
    for (uint32_t i = 0; i < s->n_regions; i++) {
        *ex++ = s->regions[i];
        EL(c, regions, s->regions[i]).parent = op;
    }
    for (uint32_t i = 0; i < s->n_succs; i++) {
        *ex++ = s->succs[i];
        *ex++ = s->n_succ_operands[i];
    }
    return op;
}

uint32_t ir_op_type(IrCtx *c, IrOp o) { return EL(c, ops, o).op_type; }
uint32_t ir_op_num_operands(IrCtx *c, IrOp o) { return EL(c, ops, o).n_operands; }
// (No nested EL(): tinyc's preprocessor does not expand a macro inside the
// argument of the same macro.)
IrValue ir_op_operand(IrCtx *c, IrOp o, uint32_t i) {
    uint32_t u = EL(c, ops, o).first_operand + i;
    return EL(c, opnds, u).value;
}
uint32_t ir_op_num_results(IrCtx *c, IrOp o) { return EL(c, ops, o).n_results; }
IrValue ir_op_result(IrCtx *c, IrOp o, uint32_t i) { return EL(c, ops, o).first_result + i; }
uint32_t ir_op_num_attrs(IrCtx *c, IrOp o) { return EL(c, ops, o).n_attrs; }
IrAttr ir_op_attr(IrCtx *c, IrOp o, uint32_t i) {
    uint32_t x = EL(c, ops, o).first_extra + i;
    return EL(c, extra, x);
}
uint32_t ir_op_num_regions(IrCtx *c, IrOp o) { return EL(c, ops, o).n_regions; }

IrRegion ir_op_region(IrCtx *c, IrOp oh, uint32_t i) {
    BOp *o = &EL(c, ops, oh);
    return EL(c, extra, o->first_extra + o->n_attrs + i);
}

uint32_t ir_op_num_succs(IrCtx *c, IrOp o) { return EL(c, ops, o).n_succs; }

IrBlock ir_op_succ(IrCtx *c, IrOp oh, uint32_t s) {
    BOp *o = &EL(c, ops, oh);
    return EL(c, extra, o->first_extra + o->n_attrs + o->n_regions + 2 * s);
}

uint32_t ir_op_num_succ_operands(IrCtx *c, IrOp oh, uint32_t s) {
    BOp *o = &EL(c, ops, oh);
    return EL(c, extra, o->first_extra + o->n_attrs + o->n_regions + 2 * s + 1);
}

IrValue ir_op_succ_operand(IrCtx *c, IrOp oh, uint32_t s, uint32_t i) {
    BOp *o = &EL(c, ops, oh);
    const uint32_t *su = &EL(c, extra, o->first_extra + o->n_attrs + o->n_regions);
    uint32_t off = o->n_operands;
    for (uint32_t k = 0; k < s; k++) off += su[2 * k + 1];
    return EL(c, opnds, o->first_operand + off + i).value;
}

IrBlock ir_op_parent_block(IrCtx *c, IrOp o) { return EL(c, ops, o).parent; }

IrType ir_value_type(IrCtx *c, IrValue v) { return EL(c, vals, v).type; }

IrOp ir_value_def_op(IrCtx *c, IrValue vh) {
    BValue *v = &EL(c, vals, vh);
    return (v->kind_index & VAL_BLOCK_ARG) ? 0 : v->def;
}

uint32_t ir_value_num_uses(IrCtx *c, IrValue v) { return EL(c, vals, v).n_uses; }
IrUse ir_value_first_use(IrCtx *c, IrValue v) { return EL(c, vals, v).first_use; }
IrUse ir_use_next(IrCtx *c, IrUse u) { return EL(c, opnds, u).next; }
IrOp ir_use_owner(IrCtx *c, IrUse u) { return EL(c, opnds, u).owner; }

uint32_t ir_use_operand_index(IrCtx *c, IrUse u) {
    BOperand *o = &EL(c, opnds, u);
    return (uint32_t)u - EL(c, ops, o->owner).first_operand;
}

void ir_op_set_operand(IrCtx *c, IrOp op, uint32_t idx, IrValue v) {
    uint32_t u = EL(c, ops, op).first_operand + idx;
    if (EL(c, opnds, u).value == v) return;
    use_unlink(c, u);
    use_push(c, v, u);
}

void ir_replace_all_uses(IrCtx *c, IrValue from, IrValue to) {
    if (from == to) return;
    BValue *ov = &EL(c, vals, from);
    BValue *nv = &EL(c, vals, to);
    uint32_t head = ov->first_use;
    if (!head) return;
    uint32_t tail = head;
    for (uint32_t u = head; u; u = EL(c, opnds, u).next) {
        EL(c, opnds, u).value = to;
        tail = u;
    }
    EL(c, opnds, tail).next = nv->first_use;
    if (nv->first_use) EL(c, opnds, nv->first_use).prev = tail;
    nv->first_use = head;
    nv->n_uses += ov->n_uses;
    ov->first_use = 0;
    ov->n_uses = 0;
}

void ir_op_erase(IrCtx *c, IrOp op) {
    BOp *o = &EL(c, ops, op);
    uint32_t first = o->first_operand;
    uint32_t n_all = o->n_operands + o->n_succ_operands;
    for (uint32_t i = 0; i < n_all; i++) use_unlink(c, first + i);
    if (o->parent) {
        BBlock *b = &EL(c, blocks, o->parent);
        uint32_t w = 0;
        for (uint32_t i = 0; i < b->n_ops; i++)
            if (b->ops[i] != op) b->ops[w++] = b->ops[i];
        b->n_ops = w;
    }
    o->parent = 0;
}

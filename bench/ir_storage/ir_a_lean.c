// Design A, compact layout: pointers into an arena, one allocation per op.
//
// An op is a 40-byte header followed by its operands, results, attribute
// handles, region handles and successor records, at offsets computed from
// the counts in the header (as upstream MLIR lays out Operation). An operand
// carries intrusive use-list links like upstream's OpOperand:
// {value, next_use, prev_next, owner} = 32 bytes. A handle is the address of
// the object: the op header, the LValue inside its op, the block, ...
//
// This is the layout the design doc proposes for the native backend (§6 N5,
// N6) with pointer handles. It isolates the pointer-vs-index question from
// the layout cleanup that design B would bring along.
#include "ir_api.h"
#include <base/arena.h>

#if IR_HANDLE_BITS != 64
#error "the A designs store pointers in handles: build with IR_HANDLE_BITS=64"
#endif
#include <string.h>

typedef struct LType { uint32_t kind; uint32_t width; } LType;   // kind 0 int, 1 ptr
typedef struct LAttr { uint32_t kind; uint32_t key; LType *type; int64_t value; } LAttr;

struct LOp;
struct LValue;

typedef struct LOperand {
    struct LValue *value;
    struct LOperand *next_use;
    struct LOperand **prev_next;   // the pointer that points at this operand
    struct LOp *owner;
} LOperand;

#define VAL_BLOCK_ARG 0x80000000u

typedef struct LValue {
    LType *type;
    void *def;                 // LOp * for a result, LBlock * for a block arg
    uint32_t kind_index;       // VAL_BLOCK_ARG | index, or the result index
    uint32_t n_uses;
    LOperand *first_use;
    void *location;
} LValue;

typedef struct LSucc { struct LBlock *block; uint32_t n_operands; uint32_t pad; } LSucc;

typedef struct LOp {
    uint16_t op_type;
    uint16_t n_attrs;
    uint16_t n_regions;
    uint16_t n_succs;
    uint32_t n_operands;       // regular operands
    uint32_t n_succ_operands;  // all successor operands
    uint32_t n_results;
    uint32_t pad;
    struct LBlock *parent;
    void *location;
} LOp;

typedef struct LBlock {
    LOp **ops;
    uint32_t n_ops, cap_ops;
    LValue **args;
    uint32_t n_args, cap_args;
    struct LRegion *parent;
} LBlock;

typedef struct LRegion {
    LBlock **blocks;
    uint32_t n_blocks, cap_blocks;
    LOp *parent;
} LRegion;

struct IrCtx {
    Arena *arena;
    uint64_t bytes;
    LType *types[16];
    uint32_t n_types;
    void *location;
};

const char *ir_impl_name(void) { return "A-lean"; }

static void *a_alloc(IrCtx *c, size_t n) {
    c->bytes += (n + 15) & ~(size_t)15;
    return arena_alloc(c->arena, n);
}

static inline LOp *OP(IrHandle h) { return (LOp *)(uintptr_t)h; }
static inline LValue *VAL(IrHandle h) { return (LValue *)(uintptr_t)h; }
static inline LBlock *BLK(IrHandle h) { return (LBlock *)(uintptr_t)h; }
static inline LRegion *REG(IrHandle h) { return (LRegion *)(uintptr_t)h; }

static inline LOperand *op_operands(LOp *o) { return (LOperand *)(o + 1); }
static inline LValue *op_results(LOp *o) {
    return (LValue *)(op_operands(o) + o->n_operands + o->n_succ_operands);
}
static inline LAttr **op_attrs(LOp *o) { return (LAttr **)(op_results(o) + o->n_results); }
static inline LRegion **op_regions(LOp *o) { return (LRegion **)(op_attrs(o) + o->n_attrs); }
static inline LSucc *op_succs(LOp *o) { return (LSucc *)(op_regions(o) + o->n_regions); }

IrCtx *ir_ctx_create(void) {
    Arena *arena = arena_create(64 * 1024 * 1024 - 64 * 1024);
    IrCtx *c = (IrCtx *)arena_alloc(arena, sizeof(IrCtx));
    memset(c, 0, sizeof(IrCtx));
    c->arena = arena;
    c->location = a_alloc(c, 32);
    return c;
}

void ir_ctx_destroy(IrCtx *c) { arena_destroy(c->arena); }
void ir_ctx_reserve(IrCtx *c, uint32_t n_ops) { (void)c; (void)n_ops; }

uint64_t ir_ctx_bytes(IrCtx *c) { return c->bytes; }

static IrType intern(IrCtx *c, uint32_t kind, uint32_t width) {
    for (uint32_t i = 0; i < c->n_types; i++)
        if (c->types[i]->kind == kind && c->types[i]->width == width)
            return (IrType)(uintptr_t)c->types[i];
    LType *t = (LType *)a_alloc(c, sizeof(LType));
    t->kind = kind;
    t->width = width;
    c->types[c->n_types++] = t;
    return (IrType)(uintptr_t)t;
}

IrType ir_type_int(IrCtx *c, uint32_t width) { return intern(c, 0, width); }
IrType ir_type_ptr(IrCtx *c) { return intern(c, 1, 0); }
uint32_t ir_type_width(IrCtx *c, IrType t) { (void)c; return ((LType *)(uintptr_t)t)->width; }

IrAttr ir_attr_int(IrCtx *c, uint32_t key, int64_t value, IrType type) {
    LAttr *a = (LAttr *)a_alloc(c, sizeof(LAttr));
    a->kind = 0;
    a->key = key;
    a->type = (LType *)(uintptr_t)type;
    a->value = value;
    return (IrAttr)(uintptr_t)a;
}

int64_t ir_attr_int_value(IrCtx *c, IrAttr a) { (void)c; return ((LAttr *)(uintptr_t)a)->value; }

IrRegion ir_region_create(IrCtx *c) {
    LRegion *r = (LRegion *)a_alloc(c, sizeof(LRegion));
    r->blocks = NULL;
    r->n_blocks = r->cap_blocks = 0;
    r->parent = NULL;
    return (IrRegion)(uintptr_t)r;
}

void ir_region_append_block(IrCtx *c, IrRegion rh, IrBlock bh) {
    LRegion *r = REG(rh);
    if (r->n_blocks == r->cap_blocks) {
        uint32_t nc = r->cap_blocks ? r->cap_blocks * 2 : 4;
        LBlock **nb = (LBlock **)a_alloc(c, nc * sizeof(LBlock *));
        if (r->n_blocks) memcpy(nb, r->blocks, r->n_blocks * sizeof(LBlock *));
        r->blocks = nb;
        r->cap_blocks = nc;
    }
    r->blocks[r->n_blocks++] = BLK(bh);
    BLK(bh)->parent = r;
}

uint32_t ir_region_num_blocks(IrCtx *c, IrRegion r) { (void)c; return REG(r)->n_blocks; }
IrBlock ir_region_block(IrCtx *c, IrRegion r, uint32_t i) { (void)c; return (IrBlock)(uintptr_t)REG(r)->blocks[i]; }

IrBlock ir_block_create(IrCtx *c) {
    LBlock *b = (LBlock *)a_alloc(c, sizeof(LBlock));
    b->ops = NULL;
    b->n_ops = b->cap_ops = 0;
    b->args = NULL;
    b->n_args = b->cap_args = 0;
    b->parent = NULL;
    return (IrBlock)(uintptr_t)b;
}

IrValue ir_block_add_arg(IrCtx *c, IrBlock bh, IrType t) {
    LBlock *b = BLK(bh);
    LValue *v = (LValue *)a_alloc(c, sizeof(LValue));
    v->type = (LType *)(uintptr_t)t;
    v->def = b;
    v->kind_index = VAL_BLOCK_ARG | b->n_args;
    v->n_uses = 0;
    v->first_use = NULL;
    v->location = c->location;
    if (b->n_args == b->cap_args) {
        uint32_t nc = b->cap_args ? b->cap_args * 2 : 4;
        LValue **na = (LValue **)a_alloc(c, nc * sizeof(LValue *));
        if (b->n_args) memcpy(na, b->args, b->n_args * sizeof(LValue *));
        b->args = na;
        b->cap_args = nc;
    }
    b->args[b->n_args++] = v;
    return (IrValue)(uintptr_t)v;
}

uint32_t ir_block_num_args(IrCtx *c, IrBlock b) { (void)c; return BLK(b)->n_args; }
IrValue ir_block_arg(IrCtx *c, IrBlock b, uint32_t i) { (void)c; return (IrValue)(uintptr_t)BLK(b)->args[i]; }
uint32_t ir_block_num_ops(IrCtx *c, IrBlock b) { (void)c; return BLK(b)->n_ops; }
IrOp ir_block_op(IrCtx *c, IrBlock b, uint32_t i) { (void)c; return (IrOp)(uintptr_t)BLK(b)->ops[i]; }

static void block_reserve(IrCtx *c, LBlock *b) {
    if (b->n_ops < b->cap_ops) return;
    uint32_t nc = b->cap_ops ? b->cap_ops * 2 : 4;
    LOp **no = (LOp **)a_alloc(c, nc * sizeof(LOp *));
    if (b->n_ops) memcpy(no, b->ops, b->n_ops * sizeof(LOp *));
    b->ops = no;
    b->cap_ops = nc;
}

void ir_block_append_op(IrCtx *c, IrBlock bh, IrOp oh) {
    LBlock *b = BLK(bh);
    block_reserve(c, b);
    b->ops[b->n_ops++] = OP(oh);
    OP(oh)->parent = b;
}

void ir_block_insert_op(IrCtx *c, IrBlock bh, uint32_t idx, IrOp oh) {
    LBlock *b = BLK(bh);
    block_reserve(c, b);
    if (idx < b->n_ops)
        memmove(b->ops + idx + 1, b->ops + idx, (b->n_ops - idx) * sizeof(LOp *));
    b->ops[idx] = OP(oh);
    b->n_ops++;
    OP(oh)->parent = b;
}

static inline void use_push(LValue *v, LOperand *u) {
    u->value = v;
    u->next_use = v->first_use;
    if (v->first_use) v->first_use->prev_next = &u->next_use;
    u->prev_next = &v->first_use;
    v->first_use = u;
    v->n_uses++;
}

static inline void use_unlink(LOperand *u) {
    LValue *v = u->value;
    if (!v) return;
    // Through a local: tinyc compiles `*u->prev_next = x` as a copy of the
    // pointed-to struct.
    LOperand **pp = u->prev_next;
    *pp = u->next_use;
    if (u->next_use) u->next_use->prev_next = u->prev_next;
    v->n_uses--;
    u->value = NULL;
    u->next_use = NULL;
    u->prev_next = NULL;
}

IrOp ir_op_create(IrCtx *c, const IrOpSpec *s) {
    uint32_t n_succ_ops = 0;
    for (uint32_t k = 0; k < s->n_succs; k++) n_succ_ops += s->n_succ_operands[k];
    uint32_t n_all = s->n_operands + n_succ_ops;
    size_t size = sizeof(LOp) + n_all * sizeof(LOperand) + s->n_results * sizeof(LValue) +
                  s->n_attrs * sizeof(LAttr *) + s->n_regions * sizeof(LRegion *) +
                  s->n_succs * sizeof(LSucc);
    LOp *o = (LOp *)a_alloc(c, size);
    o->op_type = (uint16_t)s->op_type;
    o->n_attrs = (uint16_t)s->n_attrs;
    o->n_regions = (uint16_t)s->n_regions;
    o->n_succs = (uint16_t)s->n_succs;
    o->n_operands = s->n_operands;
    o->n_succ_operands = n_succ_ops;
    o->n_results = s->n_results;
    o->pad = 0;
    o->parent = NULL;
    o->location = c->location;
    LOperand *ops = op_operands(o);
    for (uint32_t i = 0; i < n_all; i++) {
        ops[i].owner = o;
        use_push(VAL(s->operands[i]), &ops[i]);
    }
    LValue *res = op_results(o);
    for (uint32_t i = 0; i < s->n_results; i++) {
        res[i].type = (LType *)(uintptr_t)s->result_types[i];
        res[i].def = o;
        res[i].kind_index = i;
        res[i].n_uses = 0;
        res[i].first_use = NULL;
        res[i].location = c->location;
    }
    LAttr **at = op_attrs(o);
    for (uint32_t i = 0; i < s->n_attrs; i++) at[i] = (LAttr *)(uintptr_t)s->attrs[i];
    LRegion **rg = op_regions(o);
    for (uint32_t i = 0; i < s->n_regions; i++) {
        rg[i] = REG(s->regions[i]);
        rg[i]->parent = o;
    }
    LSucc *su = op_succs(o);
    for (uint32_t i = 0; i < s->n_succs; i++) {
        su[i].block = BLK(s->succs[i]);
        su[i].n_operands = s->n_succ_operands[i];
        su[i].pad = 0;
    }
    return (IrOp)(uintptr_t)o;
}

uint32_t ir_op_type(IrCtx *c, IrOp o) { (void)c; return OP(o)->op_type; }
uint32_t ir_op_num_operands(IrCtx *c, IrOp o) { (void)c; return OP(o)->n_operands; }
IrValue ir_op_operand(IrCtx *c, IrOp o, uint32_t i) { (void)c; return (IrValue)(uintptr_t)op_operands(OP(o))[i].value; }
uint32_t ir_op_num_results(IrCtx *c, IrOp o) { (void)c; return OP(o)->n_results; }
IrValue ir_op_result(IrCtx *c, IrOp o, uint32_t i) { (void)c; return (IrValue)(uintptr_t)(op_results(OP(o)) + i); }
uint32_t ir_op_num_attrs(IrCtx *c, IrOp o) { (void)c; return OP(o)->n_attrs; }
IrAttr ir_op_attr(IrCtx *c, IrOp o, uint32_t i) { (void)c; return (IrAttr)(uintptr_t)op_attrs(OP(o))[i]; }
uint32_t ir_op_num_regions(IrCtx *c, IrOp o) { (void)c; return OP(o)->n_regions; }
IrRegion ir_op_region(IrCtx *c, IrOp o, uint32_t i) { (void)c; return (IrRegion)(uintptr_t)op_regions(OP(o))[i]; }
uint32_t ir_op_num_succs(IrCtx *c, IrOp o) { (void)c; return OP(o)->n_succs; }
IrBlock ir_op_succ(IrCtx *c, IrOp o, uint32_t s) { (void)c; return (IrBlock)(uintptr_t)op_succs(OP(o))[s].block; }
uint32_t ir_op_num_succ_operands(IrCtx *c, IrOp o, uint32_t s) { (void)c; return op_succs(OP(o))[s].n_operands; }

IrValue ir_op_succ_operand(IrCtx *c, IrOp oh, uint32_t s, uint32_t i) {
    (void)c;
    LOp *o = OP(oh);
    LSucc *su = op_succs(o);
    uint32_t off = o->n_operands;
    for (uint32_t k = 0; k < s; k++) off += su[k].n_operands;
    return (IrValue)(uintptr_t)op_operands(o)[off + i].value;
}

IrBlock ir_op_parent_block(IrCtx *c, IrOp o) { (void)c; return (IrBlock)(uintptr_t)OP(o)->parent; }

IrType ir_value_type(IrCtx *c, IrValue v) { (void)c; return (IrType)(uintptr_t)VAL(v)->type; }

IrOp ir_value_def_op(IrCtx *c, IrValue vh) {
    (void)c;
    LValue *v = VAL(vh);
    return (v->kind_index & VAL_BLOCK_ARG) ? 0 : (IrOp)(uintptr_t)v->def;
}

uint32_t ir_value_num_uses(IrCtx *c, IrValue v) { (void)c; return VAL(v)->n_uses; }
IrUse ir_value_first_use(IrCtx *c, IrValue v) { (void)c; return (IrUse)(uintptr_t)VAL(v)->first_use; }
IrUse ir_use_next(IrCtx *c, IrUse u) { (void)c; return (IrUse)(uintptr_t)((LOperand *)(uintptr_t)u)->next_use; }
IrOp ir_use_owner(IrCtx *c, IrUse u) { (void)c; return (IrOp)(uintptr_t)((LOperand *)(uintptr_t)u)->owner; }

uint32_t ir_use_operand_index(IrCtx *c, IrUse uh) {
    (void)c;
    LOperand *u = (LOperand *)(uintptr_t)uh;
    return (uint32_t)(u - op_operands(u->owner));
}

void ir_op_set_operand(IrCtx *c, IrOp oh, uint32_t idx, IrValue v) {
    (void)c;
    LOperand *u = &op_operands(OP(oh))[idx];
    if (u->value == VAL(v)) return;
    use_unlink(u);
    use_push(VAL(v), u);
}

void ir_replace_all_uses(IrCtx *c, IrValue from, IrValue to) {
    (void)c;
    LValue *ov = VAL(from), *nv = VAL(to);
    if (ov == nv || !ov->first_use) return;
    LOperand *head = ov->first_use;
    LOperand *tail = head;
    for (LOperand *u = head; u; u = u->next_use) {
        u->value = nv;
        tail = u;
    }
    tail->next_use = nv->first_use;
    if (nv->first_use) nv->first_use->prev_next = &tail->next_use;
    head->prev_next = &nv->first_use;
    nv->first_use = head;
    nv->n_uses += ov->n_uses;
    ov->first_use = NULL;
    ov->n_uses = 0;
}

void ir_op_erase(IrCtx *c, IrOp oh) {
    (void)c;
    LOp *o = OP(oh);
    LOperand *ops = op_operands(o);
    uint32_t n_all = o->n_operands + o->n_succ_operands;
    for (uint32_t i = 0; i < n_all; i++) use_unlink(&ops[i]);
    LBlock *b = o->parent;
    if (b) {
        uint32_t w = 0;
        for (uint32_t i = 0; i < b->n_ops; i++)
            if (b->ops[i] != o) b->ops[w++] = b->ops[i];
        b->n_ops = w;
    }
    o->parent = NULL;
}

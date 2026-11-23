// Design A, today's layout: every IR object is its own arena allocation and a
// handle is its address. The structs, the allocation sequence of op creation,
// the def-use machinery, erase and RAUW follow mlir_api_impl.c (origin/main,
// 05969ac) with def-use tracking on, which is tinyc's configuration. Only the
// parts this benchmark does not exercise (op names, printer metadata, the
// type/attribute zoo) are left out; their fields stay so sizes match.
// Getters keep today's NULL/bounds checks and structs are zeroed with
// `= {0}` as today (tinyc stores the fields; a memset would be corec's
// byte loop). Use iteration is first/next over the UseNode list: today's
// API only has MLIR_GetValueUseOwner(idx), which walks from the head on
// every call, so the "uses" phase measures the planned iterator (§3.2).
#include "ir_api.h"
#include <base/arena.h>

#if IR_HANDLE_BITS != 64
#error "the A designs store pointers in handles: build with IR_HANDLE_BITS=64"
#endif
#include <string.h>

typedef struct { const char *str; size_t size; } string_;

typedef enum { TYPE_KIND_INTEGER = 2, TYPE_KIND_LLVM_PTR = 9 } TypeKind;

// 48 bytes, like IR_Type (its largest union member is the LLVM struct case).
typedef struct IR_Type {
    TypeKind kind;
    union {
        struct { uint32_t width; uint8_t is_signed; } integer;
        struct { uint32_t address_space; } llvm_pointer;
        struct { string_ name; uintptr_t *fields; size_t n_fields; uint8_t body_set; } llvm_struct;
    } data;
} IR_Type;

// 48 bytes, like IR_Attribute.
typedef struct IR_Attribute {
    uint32_t kind;
    union { int64_t integer_value; string_ string_value; } data;
    string_ name;
    uintptr_t type;
} IR_Attribute;

typedef struct IR_Location {
    uint32_t kind;
    string_ original_text;
} IR_Location;

typedef struct UseNode {
    struct UseNode *prev;
    struct UseNode *next;
    uintptr_t owner;
    uint16_t kind;       // 0 = regular operand, 1 = successor operand
    uint16_t succ_idx;
    uint32_t op_idx;
} UseNode;

typedef struct UseSlot { UseNode **arr; } UseSlot;

typedef struct IR_ValueAux {
    string_ register_name;
    UseNode *use_head;
    uint32_t n_uses;
} IR_ValueAux;

enum { VALUE_BLOCK_ARG = 0, VALUE_OP_RESULT = 1 };

typedef struct IR_Value {
    uint32_t kind;
    uint32_t result_index;
    uintptr_t def_handle;
    uintptr_t type;
    uintptr_t location;
    IR_ValueAux *aux;
} IR_Value;

typedef struct IR_OpAux {
    uintptr_t unnumbered_loc_def;
    string_ trailing_comment;
    string_ opname;
    int32_t source_line_start;
    UseNode **operand_uses;
    UseSlot *successor_operand_uses;
} IR_OpAux;

typedef struct IR_Op {
    uint32_t op_type;
    uint32_t n_operands;
    uintptr_t *operands;
    uintptr_t *result_types;
    uintptr_t *attributes;
    uintptr_t *regions;
    uintptr_t *successors;
    uintptr_t **successor_operands;
    uint64_t *n_successor_operands;
    uintptr_t *results;
    uintptr_t location;
    IR_OpAux *aux;
    uintptr_t parent_block;
    uint32_t n_result_types;
    uint32_t n_attributes;
    uint32_t n_regions;
    uint32_t n_successors;
    uint32_t n_results;
} IR_Op;

typedef struct IR_Block {
    uintptr_t *operations;
    uintptr_t *arguments;
    uintptr_t parent_region;
    uint32_t n_operations;
    uint32_t cap_operations;
    uint32_t n_arguments;
    uint32_t cap_arguments;
} IR_Block;

typedef struct IR_Region {
    uintptr_t *blocks;
    uint64_t n_blocks;
    uint64_t cap_blocks;
} IR_Region;

struct IrCtx {
    Arena *arena;
    uint64_t bytes;
    // mlir_api_impl.c keeps these two registries in process-wide statics;
    // the cost is the same.
    uintptr_t *all_ops;
    size_t n_all_ops, cap_all_ops;
    uintptr_t *type_handles;
    size_t n_types, cap_types;
    uintptr_t location;   // one shared unknown location
};

const char *ir_impl_name(void) { return "A-today"; }

static void *a_alloc(IrCtx *c, size_t n) {
    c->bytes += (n + 15) & ~(size_t)15;
    return arena_alloc(c->arena, n);
}
#define A_NEW(c, T) ((T *)a_alloc((c), sizeof(T)))
#define A_NEW_ARRAY(c, T, n) ((T *)a_alloc((c), sizeof(T) * (size_t)(n)))

static inline IR_Op *resolve_op(uintptr_t h) { return (IR_Op *)h; }
static inline IR_Block *resolve_block(uintptr_t h) { return (IR_Block *)h; }
static inline IR_Region *resolve_region(uintptr_t h) { return (IR_Region *)h; }
static inline IR_Value *resolve_value(uintptr_t h) { return (IR_Value *)h; }
static inline IR_Type *resolve_type(uintptr_t h) { return (IR_Type *)h; }
static inline IR_Attribute *resolve_attr(uintptr_t h) { return (IR_Attribute *)h; }

IrCtx *ir_ctx_create(void) {
    Arena *arena = arena_create(64 * 1024 * 1024 - 64 * 1024);
    IrCtx *c = (IrCtx *)arena_alloc(arena, sizeof(IrCtx));
    memset(c, 0, sizeof(IrCtx));
    c->arena = arena;
    IR_Location *l = A_NEW(c, IR_Location);
    memset(l, 0, sizeof(IR_Location));
    c->location = (uintptr_t)l;
    return c;
}

void ir_ctx_destroy(IrCtx *c) { arena_destroy(c->arena); }

void ir_ctx_reserve(IrCtx *c, uint32_t n_ops) { (void)c; (void)n_ops; }

uint64_t ir_ctx_bytes(IrCtx *c) { return c->bytes; }

// --- def-use machinery (mlir_api_impl.c use_list_* / use_register_*) ---

static inline void use_list_unlink(IR_Value *v, UseNode *u) {
    if (!v || !v->aux || !u) return;
    if (u->prev) u->prev->next = u->next;
    else         v->aux->use_head = u->next;
    if (u->next) u->next->prev = u->prev;
    u->prev = u->next = NULL;
    if (v->aux->n_uses) v->aux->n_uses--;
}

static inline void use_list_push_front(IR_Value *v, UseNode *u) {
    if (!v || !v->aux || !u) return;
    u->prev = NULL;
    u->next = v->aux->use_head;
    if (v->aux->use_head) v->aux->use_head->prev = u;
    v->aux->use_head = u;
    v->aux->n_uses++;
}

static UseNode *use_register(IrCtx *c, uintptr_t owner, uintptr_t value,
                             uint16_t kind, uint32_t succ_idx, uint32_t op_idx) {
    if (value == 0) return NULL;
    IR_Value *v = resolve_value(value);
    UseNode *u = A_NEW(c, UseNode);
    u->owner = owner;
    u->kind = kind;
    u->succ_idx = (uint16_t)succ_idx;
    u->op_idx = op_idx;
    use_list_push_front(v, u);
    return u;
}

static inline void use_unregister(uintptr_t value, UseNode *u) {
    if (value == 0 || !u) return;
    use_list_unlink(resolve_value(value), u);
}

static void register_op(IrCtx *c, uintptr_t h) {
    if (c->n_all_ops == c->cap_all_ops) {
        size_t nc = c->cap_all_ops ? c->cap_all_ops * 2 : 256;
        uintptr_t *p = A_NEW_ARRAY(c, uintptr_t, nc);
        if (c->n_all_ops) memcpy(p, c->all_ops, c->n_all_ops * sizeof(uintptr_t));
        c->all_ops = p;
        c->cap_all_ops = nc;
    }
    c->all_ops[c->n_all_ops++] = h;
}

// --- types and attributes ---

static int type_eq(const IR_Type *a, const IR_Type *b) {
    if (a->kind != b->kind) return 0;
    if (a->kind == TYPE_KIND_INTEGER)
        return a->data.integer.width == b->data.integer.width &&
               a->data.integer.is_signed == b->data.integer.is_signed;
    return a->data.llvm_pointer.address_space == b->data.llvm_pointer.address_space;
}

static IrType intern_type(IrCtx *c, IR_Type t) {
    for (size_t i = 0; i < c->n_types; i++)
        if (type_eq(resolve_type(c->type_handles[i]), &t)) return c->type_handles[i];
    IR_Type *slot = A_NEW(c, IR_Type);
    *slot = t;
    if (c->n_types == c->cap_types) {
        size_t nc = c->cap_types ? c->cap_types * 2 : 64;
        uintptr_t *nh = A_NEW_ARRAY(c, uintptr_t, nc);
        if (c->n_types) memcpy(nh, c->type_handles, c->n_types * sizeof(uintptr_t));
        c->type_handles = nh;
        c->cap_types = nc;
    }
    c->type_handles[c->n_types++] = (uintptr_t)slot;
    return (uintptr_t)slot;
}

IrType ir_type_int(IrCtx *c, uint32_t width) {
    IR_Type t;
    memset(&t, 0, sizeof(t));
    t.kind = TYPE_KIND_INTEGER;
    t.data.integer.width = width;
    return intern_type(c, t);
}

IrType ir_type_ptr(IrCtx *c) {
    IR_Type t;
    memset(&t, 0, sizeof(t));
    t.kind = TYPE_KIND_LLVM_PTR;
    return intern_type(c, t);
}

uint32_t ir_type_width(IrCtx *c, IrType th) {
    (void)c;
    IR_Type *t = resolve_type(th);
    return t->kind == TYPE_KIND_INTEGER ? t->data.integer.width : 0;
}

// The callers pass str_lit("value") etc., whose length is a constant.
static string_ attr_name(uint32_t key) {
    string_ s;
    switch (key) {
        case ATTR_VALUE: s.str = "value"; s.size = 5; break;
        case ATTR_PREDICATE: s.str = "predicate"; s.size = 9; break;
        case ATTR_CALLEE: s.str = "callee"; s.size = 6; break;
        case ATTR_ELEM_TYPE: s.str = "elem_type"; s.size = 9; break;
        case ATTR_SYM_NAME: s.str = "sym_name"; s.size = 8; break;
        default: s.str = "function_type"; s.size = 13; break;
    }
    return s;
}

IrAttr ir_attr_int(IrCtx *c, uint32_t key, int64_t value, IrType type) {
    // MLIR_CreateAttributeInteger: copy the name into the arena
    // (arena_dup_str), then allocate the attribute.
    string_ nm = attr_name(key);
    char *p = A_NEW_ARRAY(c, char, nm.size);
    memcpy(p, nm.str, nm.size);
    IR_Attribute a = {0};
    a.kind = 0;
    a.data.integer_value = value;
    a.name.str = p;
    a.name.size = nm.size;
    a.type = type;
    IR_Attribute *slot = A_NEW(c, IR_Attribute);
    *slot = a;
    return (uintptr_t)slot;
}

int64_t ir_attr_int_value(IrCtx *c, IrAttr ah) {
    (void)c;
    return resolve_attr(ah)->data.integer_value;
}

// --- regions and blocks ---

IrRegion ir_region_create(IrCtx *c) {
    IR_Region r = {0};
    IR_Region *slot = A_NEW(c, IR_Region);
    *slot = r;
    return (uintptr_t)slot;
}

void ir_region_append_block(IrCtx *c, IrRegion rh, IrBlock block) {
    IR_Region *region = resolve_region(rh);
    if (region->n_blocks >= region->cap_blocks) {
        uint64_t new_cap = region->cap_blocks ? region->cap_blocks * 2 : 4;
        uintptr_t *nb = A_NEW_ARRAY(c, uintptr_t, new_cap);
        if (region->blocks) memcpy(nb, region->blocks, region->n_blocks * sizeof(uintptr_t));
        region->blocks = nb;
        region->cap_blocks = new_cap;
    }
    region->blocks[region->n_blocks++] = block;
    resolve_block(block)->parent_region = rh;
}

uint32_t ir_region_num_blocks(IrCtx *c, IrRegion rh) {
    (void)c;
    IR_Region *r = resolve_region(rh);
    return r ? (uint32_t)r->n_blocks : 0;
}

IrBlock ir_region_block(IrCtx *c, IrRegion rh, uint32_t i) {
    (void)c;
    IR_Region *r = resolve_region(rh);
    if (!r || i >= r->n_blocks) return 0;
    return r->blocks[i];
}

IrBlock ir_block_create(IrCtx *c) {
    IR_Block b = {0};
    IR_Block *slot = A_NEW(c, IR_Block);
    *slot = b;
    return (uintptr_t)slot;
}

static IR_ValueAux *value_aux_create(IrCtx *c) {
    IR_ValueAux *a = A_NEW(c, IR_ValueAux);
    *a = (IR_ValueAux){0};
    return a;
}

static uintptr_t alloc_value(IrCtx *c, uint32_t kind, uintptr_t def, uint32_t idx, uintptr_t type) {
    IR_Value v = {0};
    v.kind = kind;
    v.aux = value_aux_create(c);
    v.result_index = idx;
    v.type = type;
    v.location = c->location;
    v.def_handle = def;
    IR_Value *slot = A_NEW(c, IR_Value);
    *slot = v;
    return (uintptr_t)slot;
}

IrValue ir_block_add_arg(IrCtx *c, IrBlock bh, IrType t) {
    IR_Block *block = resolve_block(bh);
    // MLIR_CreateValueBlockArg + MLIR_AppendBlockArg.
    uintptr_t arg = alloc_value(c, VALUE_BLOCK_ARG, 0, block->n_arguments, t);
    if (block->n_arguments >= block->cap_arguments) {
        uint32_t new_cap = block->cap_arguments ? block->cap_arguments * 2 : 4;
        uintptr_t *na = A_NEW_ARRAY(c, uintptr_t, new_cap);
        if (block->arguments) memcpy(na, block->arguments, block->n_arguments * sizeof(uintptr_t));
        block->arguments = na;
        block->cap_arguments = new_cap;
    }
    block->arguments[block->n_arguments++] = arg;
    resolve_value(arg)->def_handle = bh;
    return arg;
}

uint32_t ir_block_num_args(IrCtx *c, IrBlock bh) {
    (void)c; IR_Block *b = resolve_block(bh); return b ? b->n_arguments : 0;
}
IrValue ir_block_arg(IrCtx *c, IrBlock bh, uint32_t i) {
    (void)c; IR_Block *b = resolve_block(bh);
    if (!b || i >= b->n_arguments) return 0;
    return b->arguments[i];
}
uint32_t ir_block_num_ops(IrCtx *c, IrBlock bh) {
    (void)c; IR_Block *b = resolve_block(bh); return b ? b->n_operations : 0;
}
IrOp ir_block_op(IrCtx *c, IrBlock bh, uint32_t i) {
    (void)c; IR_Block *b = resolve_block(bh);
    if (!b || i >= b->n_operations) return 0;
    return b->operations[i];
}

void ir_block_append_op(IrCtx *c, IrBlock bh, IrOp op) {
    IR_Block *block = resolve_block(bh);
    if (block->n_operations >= block->cap_operations) {
        uint32_t new_cap = block->cap_operations ? block->cap_operations * 2 : 4;
        uintptr_t *no = A_NEW_ARRAY(c, uintptr_t, new_cap);
        if (block->operations) memcpy(no, block->operations, block->n_operations * sizeof(uintptr_t));
        block->operations = no;
        block->cap_operations = new_cap;
    }
    block->operations[block->n_operations++] = op;
    resolve_op(op)->parent_block = bh;
}

void ir_block_insert_op(IrCtx *c, IrBlock bh, uint32_t idx, IrOp op) {
    IR_Block *block = resolve_block(bh);
    if (idx > block->n_operations) idx = block->n_operations;
    if (block->n_operations >= block->cap_operations) {
        uint32_t new_cap = block->cap_operations ? block->cap_operations * 2 : 4;
        uintptr_t *no = A_NEW_ARRAY(c, uintptr_t, new_cap);
        if (block->operations && idx > 0)
            memcpy(no, block->operations, idx * sizeof(uintptr_t));
        if (block->operations && idx < block->n_operations)
            memcpy(no + idx + 1, block->operations + idx,
                   (block->n_operations - idx) * sizeof(uintptr_t));
        block->operations = no;
        block->cap_operations = new_cap;
    } else if (idx < block->n_operations) {
        memmove(block->operations + idx + 1, block->operations + idx,
                (block->n_operations - idx) * sizeof(uintptr_t));
    }
    block->operations[idx] = op;
    block->n_operations++;
    resolve_op(op)->parent_block = bh;
}

// --- ops ---

static uintptr_t *dup_handles(IrCtx *c, const IrHandle *src, uint32_t n) {
    if (n == 0) return NULL;
    uintptr_t *d = A_NEW_ARRAY(c, uintptr_t, n);
    memcpy(d, src, n * sizeof(uintptr_t));
    return d;
}

IrOp ir_op_create(IrCtx *c, const IrOpSpec *s) {
    // The caller-side half of op creation in tinyc: one
    // MLIR_CreateValueOpResult per result before MLIR_CreateOp.
    uintptr_t results[8];
    for (uint32_t i = 0; i < s->n_results; i++)
        results[i] = alloc_value(c, VALUE_OP_RESULT, 0, i, s->result_types[i]);

    // MLIR_CreateOpWithSuccessors.
    IR_Op op = {0};
    op.op_type = s->op_type;
    op.attributes = dup_handles(c, s->attrs, s->n_attrs);
    op.n_attributes = s->n_attrs;
    op.result_types = dup_handles(c, s->result_types, s->n_results);
    op.n_result_types = s->n_results;
    op.results = dup_handles(c, results, s->n_results);
    op.n_results = s->n_results;
    op.operands = dup_handles(c, s->operands, s->n_operands);
    op.n_operands = s->n_operands;
    op.regions = dup_handles(c, s->regions, s->n_regions);
    op.n_regions = s->n_regions;
    op.successors = dup_handles(c, s->succs, s->n_succs);
    op.n_successors = s->n_succs;
    IR_OpAux *aux = A_NEW(c, IR_OpAux);
    *aux = (IR_OpAux){0};
    aux->source_line_start = -1;
    op.aux = aux;
    const IrHandle *succ_src = s->operands + s->n_operands;
    if (s->n_succs > 0) {
        op.successor_operands = A_NEW_ARRAY(c, uintptr_t *, s->n_succs);
        op.n_successor_operands = A_NEW_ARRAY(c, uint64_t, s->n_succs);
        aux->successor_operand_uses = A_NEW_ARRAY(c, UseSlot, s->n_succs);
        for (uint32_t k = 0; k < s->n_succs; k++) {
            uint32_t cnt = s->n_succ_operands[k];
            op.n_successor_operands[k] = cnt;
            if (cnt > 0) {
                op.successor_operands[k] = A_NEW_ARRAY(c, uintptr_t, cnt);
                memcpy(op.successor_operands[k], succ_src, cnt * sizeof(uintptr_t));
                aux->successor_operand_uses[k].arr = A_NEW_ARRAY(c, UseNode *, cnt);
                for (uint32_t j = 0; j < cnt; j++) aux->successor_operand_uses[k].arr[j] = NULL;
            } else {
                op.successor_operands[k] = NULL;
                aux->successor_operand_uses[k].arr = NULL;
            }
            succ_src += cnt;
        }
    }
    if (s->n_operands > 0) {
        aux->operand_uses = A_NEW_ARRAY(c, UseNode *, s->n_operands);
        for (uint32_t i = 0; i < s->n_operands; i++) aux->operand_uses[i] = NULL;
    }
    op.location = c->location;

    IR_Op *slot = A_NEW(c, IR_Op);
    *slot = op;
    uintptr_t handle = (uintptr_t)slot;
    register_op(c, handle);

    for (uint32_t i = 0; i < slot->n_operands; i++)
        aux->operand_uses[i] = use_register(c, handle, slot->operands[i], 0, 0, i);
    for (uint32_t k = 0; k < slot->n_successors; k++) {
        uint64_t cnt = slot->n_successor_operands[k];
        for (uint64_t j = 0; j < cnt; j++)
            aux->successor_operand_uses[k].arr[j] = use_register(
                c, handle, slot->successor_operands[k][j], 1, k, (uint32_t)j);
    }
    for (uint32_t i = 0; i < s->n_results; i++) {
        IR_Value *v = resolve_value(results[i]);
        v->def_handle = handle;
    }
    return handle;
}

// Getters keep mlir_api_impl.c's NULL and bounds checks.
uint32_t ir_op_type(IrCtx *c, IrOp oh) {
    (void)c; IR_Op *op = resolve_op(oh); return op ? op->op_type : 0;
}
uint32_t ir_op_num_operands(IrCtx *c, IrOp oh) {
    (void)c; IR_Op *op = resolve_op(oh); return op ? op->n_operands : 0;
}
IrValue ir_op_operand(IrCtx *c, IrOp oh, uint32_t i) {
    (void)c; IR_Op *op = resolve_op(oh);
    if (!op || i >= op->n_operands) return 0;
    return op->operands[i];
}
uint32_t ir_op_num_results(IrCtx *c, IrOp oh) {
    (void)c; IR_Op *op = resolve_op(oh); return op ? op->n_results : 0;
}
IrValue ir_op_result(IrCtx *c, IrOp oh, uint32_t i) {
    (void)c; IR_Op *op = resolve_op(oh);
    if (!op || i >= op->n_results) return 0;
    return op->results[i];
}
uint32_t ir_op_num_attrs(IrCtx *c, IrOp oh) {
    (void)c; IR_Op *op = resolve_op(oh); return op ? op->n_attributes : 0;
}
IrAttr ir_op_attr(IrCtx *c, IrOp oh, uint32_t i) {
    (void)c; IR_Op *op = resolve_op(oh);
    if (!op || i >= op->n_attributes) return 0;
    return op->attributes[i];
}
uint32_t ir_op_num_regions(IrCtx *c, IrOp oh) {
    (void)c; IR_Op *op = resolve_op(oh); return op ? op->n_regions : 0;
}
IrRegion ir_op_region(IrCtx *c, IrOp oh, uint32_t i) {
    (void)c; IR_Op *op = resolve_op(oh);
    if (!op || i >= op->n_regions) return 0;
    return op->regions[i];
}
uint32_t ir_op_num_succs(IrCtx *c, IrOp oh) {
    (void)c; IR_Op *op = resolve_op(oh); return op ? op->n_successors : 0;
}
IrBlock ir_op_succ(IrCtx *c, IrOp oh, uint32_t s) {
    (void)c; IR_Op *op = resolve_op(oh);
    if (!op || s >= op->n_successors) return 0;
    return op->successors[s];
}
uint32_t ir_op_num_succ_operands(IrCtx *c, IrOp oh, uint32_t s) {
    (void)c; IR_Op *op = resolve_op(oh);
    if (!op || s >= op->n_successors || !op->n_successor_operands) return 0;
    return (uint32_t)op->n_successor_operands[s];
}
IrValue ir_op_succ_operand(IrCtx *c, IrOp oh, uint32_t s, uint32_t i) {
    (void)c; IR_Op *op = resolve_op(oh);
    if (!op || s >= op->n_successors || !op->successor_operands) return 0;
    if (i >= op->n_successor_operands[s]) return 0;
    return op->successor_operands[s][i];
}
IrBlock ir_op_parent_block(IrCtx *c, IrOp oh) { (void)c; return resolve_op(oh)->parent_block; }

IrType ir_value_type(IrCtx *c, IrValue vh) {
    (void)c; IR_Value *v = resolve_value(vh); return v ? v->type : 0;
}

IrOp ir_value_def_op(IrCtx *c, IrValue vh) {
    (void)c;
    IR_Value *v = resolve_value(vh);
    return (v && v->kind == VALUE_OP_RESULT) ? v->def_handle : 0;
}

uint32_t ir_value_num_uses(IrCtx *c, IrValue vh) {
    (void)c;
    IR_Value *v = resolve_value(vh);
    return (v && v->aux) ? v->aux->n_uses : 0;
}

IrUse ir_value_first_use(IrCtx *c, IrValue vh) {
    (void)c;
    IR_Value *v = resolve_value(vh);
    return (IrUse)(uintptr_t)(v->aux ? v->aux->use_head : NULL);
}

IrUse ir_use_next(IrCtx *c, IrUse u) { (void)c; return (IrUse)(uintptr_t)((UseNode *)(uintptr_t)u)->next; }
IrOp ir_use_owner(IrCtx *c, IrUse u) { (void)c; return ((UseNode *)(uintptr_t)u)->owner; }

uint32_t ir_use_operand_index(IrCtx *c, IrUse uh) {
    (void)c;
    UseNode *u = (UseNode *)(uintptr_t)uh;
    if (u->kind == 0) return u->op_idx;
    IR_Op *o = resolve_op(u->owner);
    uint32_t off = o->n_operands;
    for (uint16_t s = 0; s < u->succ_idx; s++) off += (uint32_t)o->n_successor_operands[s];
    return off + u->op_idx;
}

void ir_op_set_operand(IrCtx *c, IrOp op, uint32_t idx, IrValue value) {
    IR_Op *o = resolve_op(op);
    if (idx < o->n_operands) {
        uintptr_t old = o->operands[idx];
        if (old == value) return;
        use_unregister(old, o->aux->operand_uses[idx]);
        o->aux->operand_uses[idx] = NULL;
        o->operands[idx] = value;
        o->aux->operand_uses[idx] = use_register(c, op, value, 0, 0, idx);
        return;
    }
    uint32_t rem = idx - o->n_operands;
    for (uint32_t s = 0; s < o->n_successors; s++) {
        uint32_t cnt = (uint32_t)o->n_successor_operands[s];
        if (rem < cnt) {
            uintptr_t *seg = o->successor_operands[s];
            uintptr_t old = seg[rem];
            if (old == value) return;
            use_unregister(old, o->aux->successor_operand_uses[s].arr[rem]);
            o->aux->successor_operand_uses[s].arr[rem] = NULL;
            seg[rem] = value;
            o->aux->successor_operand_uses[s].arr[rem] = use_register(c, op, value, 1, s, rem);
            return;
        }
        rem -= cnt;
    }
}

void ir_replace_all_uses(IrCtx *c, IrValue old_value, IrValue new_value) {
    (void)c;
    if (old_value == new_value) return;
    IR_Value *ov = resolve_value(old_value);
    if (!ov->aux || !ov->aux->use_head) return;
    IR_Value *nv = resolve_value(new_value);
    UseNode *u = ov->aux->use_head;
    while (u) {
        UseNode *next = u->next;
        IR_Op *o = resolve_op(u->owner);
        if (u->kind == 0 && u->op_idx < o->n_operands)
            o->operands[u->op_idx] = new_value;
        else if (u->kind == 1 && u->succ_idx < o->n_successors &&
                 u->op_idx < o->n_successor_operands[u->succ_idx])
            o->successor_operands[u->succ_idx][u->op_idx] = new_value;
        u = next;
    }
    UseNode *head = ov->aux->use_head;
    UseNode *tail = head;
    while (tail->next) tail = tail->next;
    tail->next = nv->aux->use_head;
    if (nv->aux->use_head) nv->aux->use_head->prev = tail;
    nv->aux->use_head = head;
    nv->aux->n_uses += ov->aux->n_uses;
    ov->aux->use_head = NULL;
    ov->aux->n_uses = 0;
}

void ir_op_erase(IrCtx *c, IrOp op) {
    (void)c;
    IR_Op *o = resolve_op(op);
    if (o->aux && o->aux->operand_uses) {
        for (uint32_t i = 0; i < o->n_operands; i++) {
            use_unregister(o->operands[i], o->aux->operand_uses[i]);
            o->aux->operand_uses[i] = NULL;
        }
    }
    if (o->aux && o->aux->successor_operand_uses) {
        for (uint32_t s = 0; s < o->n_successors; s++) {
            if (!o->aux->successor_operand_uses[s].arr) continue;
            uint64_t cnt = o->n_successor_operands[s];
            for (uint64_t k = 0; k < cnt; k++) {
                use_unregister(o->successor_operands[s][k], o->aux->successor_operand_uses[s].arr[k]);
                o->aux->successor_operand_uses[s].arr[k] = NULL;
            }
        }
    }
    IR_Block *b = resolve_block(o->parent_block);
    if (b) {
        uint32_t w = 0;
        for (uint32_t i = 0; i < b->n_operations; i++)
            if (b->operations[i] != op) b->operations[w++] = b->operations[i];
        b->n_operations = w;
    }
    o->parent_block = 0;
}

#include <stdio.h>
#include <base/arena.h>
#include <base/string.h>
#include <base/strbuf.h>
#include <base/format.h>
#include <base/hashtable.h>
#include <base/vector.h>
#include <base/assert.h>
#include <base/buddy.h>
#include "tokenizer.h"
#include "mlir_api.h"
#include "mlir_op_names.h"
#include "mlir_parser.h"
#include "mlir_printer.h"
#include "mlir_passes.h"
#include "mlir_api_upstream.h"
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Contract checks (mlir_api.h, "Diagnostics"). With -DMLIR_API_CHECKS a
// violated precondition aborts with the API function's name; without it the
// checks compile to nothing and the hot paths are unchanged.
#ifdef MLIR_API_CHECKS
static void api_check_failed(const char *fn, const char *what) {
    fprintf(stderr, "MLIR API contract violation in %s: %s\n", fn, what);
    abort();
}
#define API_CHECK(cond, fn, what) do { if (!(cond)) api_check_failed(fn, what); } while (0)
#else
#define API_CHECK(cond, fn, what) ((void)0)
#endif

// Internal concrete IR definitions

typedef enum {
    TYPE_KIND_UNKNOWN,
    TYPE_KIND_OPAQUE,
    TYPE_KIND_INTEGER,
    TYPE_KIND_FLOAT,
    TYPE_KIND_MEMREF,
    TYPE_KIND_TENSOR,
    TYPE_KIND_FUNCTION,
    TYPE_KIND_INDEX,
    TYPE_KIND_POINTER,
    TYPE_KIND_LLVM_PTR,
    TYPE_KIND_LLVM_VOID,
    TYPE_KIND_LLVM_ARRAY,
    TYPE_KIND_LLVM_STRUCT,
    TYPE_KIND_LLVM_FUNCTION
} TypeKind;

typedef enum {
    LOC_KIND_UNKNOWN,
    LOC_KIND_FILE,
    LOC_KIND_NAME,
    LOC_KIND_CALLSITE,
    LOC_KIND_FUSED,
    LOC_KIND_REF
} LocationKind;

// Internal struct definitions using handle fields

typedef struct IR_Type {
    TypeKind kind;
    union {
        struct {
            uint32_t width;
            bool is_signed;
        } integer;
        struct {
            uint32_t width;
            bool is_bfloat;
        } floating;
        struct {
            MLIR_TypeHandle element_type;
            int64_t *shape;
            uint32_t rank;
        } shaped;
        struct {
            MLIR_TypeHandle element_type;
            uint32_t address_space;
            bool has_address_space;
        } pointer;
        struct {
            MLIR_TypeHandle *inputs;
            size_t n_inputs;
            MLIR_TypeHandle *results;
            size_t n_results;
        } function;
        struct {
            MLIR_TypeHandle element;
            uint64_t count;
        } llvm_array;
        struct {
            uint32_t address_space;
        } llvm_pointer;
        struct {
            // Anonymous structs have name.size == 0. Identified structs are
            // interned by name (see intern_llvm_struct in this file).
            string name;
            MLIR_TypeHandle *fields;
            size_t n_fields;
            bool body_set;
        } llvm_struct;
        struct {
            // LLVM function type: exactly one return type (LLVM void marker
            // when the function returns void), N input types, optional varargs.
            MLIR_TypeHandle result;
            MLIR_TypeHandle *inputs;
            size_t n_inputs;
            bool is_var_arg;
        } llvm_function;
    } data;
} IR_Type;

typedef struct IR_Attribute {
    enum {
        ATTR_KIND_INTEGER,
        ATTR_KIND_FLOAT,
        ATTR_KIND_STRING,
        ATTR_KIND_BOOL,
        ATTR_KIND_ARRAY,
        ATTR_KIND_DICT,
        ATTR_KIND_TYPE
    } kind;
    union {
        int64_t integer_value;
        double float_value;
        string string_value;
        bool bool_value;
        struct {
            MLIR_AttributeHandle *elements;
            size_t count;
        } array;
        MLIR_TypeHandle type_value;
    } data;
    string name;
    // MLIR type of the attribute's value. Required for ATTR_KIND_INTEGER
    // and ATTR_KIND_FLOAT (and will apply to ATTR_KIND_DENSE_ELEMENTS
    // when added). For other kinds it is unused and must be
    // MLIR_INVALID_HANDLE (the zero value, set by `IR_Attribute a = {0}`).
    MLIR_TypeHandle type;
} IR_Attribute;

typedef struct IR_Location {
    LocationKind kind;
    union {
        struct { string filename; int line; int column; } file;
        struct { string name; } name;
        struct { int ref_id; } ref;
    } data;
    string original_text;
} IR_Location;

// --- IR storage (doc/mlir_api_design.md section 2.2, option D; #354) --------
//
// Ops, values, blocks and regions live in context-owned vectors, one per
// kind, and a handle is the element's 32-bit index (0 is the invalid handle,
// so element 0 of every vector is a reserved, zeroed record). The variable
// parts of an op live in shared vectors too:
//   - opnds: one 16-byte record {value, owner, next, prev} per operand, all
//     indices. An op's regular operands and then its successor operands (D5)
//     are consecutive, and the op stores the first one. The def-use lists are
//     threaded through these records, so a use is an operand index.
//   - vals: an op's results are consecutive, so result i is first_result + i.
//   - extra: 32-bit words; an op's region indices, then a (block, operand
//     count) pair per successor.
//   - ptrs: pointer-sized words; an op's attribute handles (attributes are
//     arena pointers until they are interned as indices, #350), and the
//     explicit result types of the rare op whose results do not carry them.
// A block's op and argument lists and a region's block list are arrays of
// 32-bit indices in the IR arena. Erased ops stay as empty slots: there is no
// free list and no compaction.
//
// Every vector is one flat array. The context reserves them from a size hint
// (MLIR_ContextOptions.expected_ops, MLIR_SetContextExpectedOps); when one
// fills up, vec_reserve doubles it and copies, so an index stays valid but a
// pointer into a vector does not. Rule: no pointer into a vector is held
// across a call that can create IR; re-resolve the index instead. Under
// MLIR_API_CHECKS each vector counts its moves (`gen`) and the functions that
// hold a pointer across an allocation check that it did not move.

typedef struct IR_Vec {
    void *data;
    uint32_t n;      // elements in use, including the reserved element 0
    uint32_t cap;    // elements allocated
    uint32_t elem;   // element size in bytes
    uint32_t gen;    // incremented whenever `data` moves
} IR_Vec;

typedef struct IR_Operand {
    uint32_t value;  // value index, 0 = none
    uint32_t owner;  // op index
    uint32_t next;   // next use of `value` (operand index), 0 = end
    uint32_t prev;   // previous use, 0 = this is the value's first use
} IR_Operand;

// IR_Value.index: the result or argument index, plus these flags.
#define VAL_BLOCK_ARG 0x80000000u  // a block argument (else an op result)
#define VAL_TRACKED   0x40000000u  // has a use list (see value_create)
#define VAL_INDEX_MASK 0x3fffffffu

typedef struct IR_Value {
    MLIR_TypeHandle type;
    MLIR_LocationHandle location;
    uint32_t def;        // op index (result) or block index (block argument)
    uint32_t index;      // result/argument index | VAL_* flags
    uint32_t first_use;  // operand index, 0 = no uses
    uint32_t n_uses;
    uint32_t name;       // register name: index into ctx->names, 0 = none
    uint32_t pad;
} IR_Value;

// IR_Op.flags
#define OPF_TRACKED  1u  // operands were linked into use lists at creation

typedef struct IR_Op {
    MLIR_OpType op_type;
    uint32_t parent_block;    // block index, 0 = not in a block
    uint32_t first_operand;   // opnds index
    uint32_t n_operands;      // regular operands
    uint32_t n_succ_operands; // all successor operands, after the regular ones
    uint32_t first_result;    // vals index
    uint32_t n_results;
    uint32_t first_extra;     // extra index: regions, then (block, count) pairs
    uint16_t n_regions;
    uint16_t n_successors;
    uint32_t first_attr;      // ptrs index
    uint16_t n_attributes;
    uint16_t flags;
    uint32_t aux;             // op_aux index, 0 = none
    MLIR_LocationHandle location;
} IR_Op;

// Rarely-used op metadata consulted only by the classic printer and the
// parser roundtrip (trailing comments, the unnumbered location def, the
// recorded source line), plus the result types of an op whose results do not
// carry them (MLIR_CreateOp with n_result_types != n_results).
typedef struct IR_OpAux {
    MLIR_LocationHandle unnumbered_loc_def;
    string trailing_comment;
    int32_t source_line_start;
    uint32_t first_result_type;  // ptrs index
    uint32_t n_result_types;
    bool explicit_result_types;
} IR_OpAux;

typedef struct IR_Block {
    uint32_t *ops;
    uint32_t *args;
    uint32_t n_ops, cap_ops;
    uint32_t n_args, cap_args;
    uint32_t parent_region;  // region index, 0 = detached
    uint32_t pad;
} IR_Block;

typedef struct IR_Region {
    uint32_t *blocks;
    uint32_t n_blocks, cap_blocks;
    // The op whose region this is, or 0 while the region is detached
    // (MLIR_CreateRegion, MLIR_TakeOpRegion). Set by op creation.
    uint32_t parent_op;
    uint32_t pad;
} IR_Region;

// The context. Everything that is not IR lives here; there is no
// process-wide state. Types (and the arrays they point to) are allocated in
// `persistent`, which the context owns, so they outlive every IR arena.
struct MLIR_Context {
    Arena *arena;          // current IR arena (owned by the caller)
    Arena *persistent;     // owned: this struct, types, the interning tables
    bool no_def_use_tracking;
    // The IR vectors (see above), owned by the context.
    IR_Vec ops, vals, opnds, extra, ptrs, blocks, regions, op_aux, names;
    // Uniqued types, compared structurally by intern_type.
    MLIR_TypeHandle *type_handles;
    size_t n_types;
    size_t cap_types;
    // Identified LLVM structs, uniqued by name.
    MLIR_TypeHandle *struct_handles;
    string *struct_names;
    size_t n_structs;
    size_t cap_structs;
    // Diagnostic handler; NULL means the default (print to stderr).
    MLIR_DiagnosticHandler diag_fn;
    void *diag_user;
    // Op types: the names of the static and dynamic types and the reverse
    // map (MLIR_RegisterOpType).
    OpTypeTable op_types;
};

#define OPS(c)     ((IR_Op *)(c)->ops.data)
#define VALS(c)    ((IR_Value *)(c)->vals.data)
#define OPNDS(c)   ((IR_Operand *)(c)->opnds.data)
#define EXTRA(c)   ((uint32_t *)(c)->extra.data)
#define PTRS(c)    ((uintptr_t *)(c)->ptrs.data)
#define BLOCKS(c)  ((IR_Block *)(c)->blocks.data)
#define REGIONS(c) ((IR_Region *)(c)->regions.data)
#define OPAUX(c)   ((IR_OpAux *)(c)->op_aux.data)
#define NAMES(c)   ((string *)(c)->names.data)

// The one growth hook: makes room for `need` elements. The capacity at least
// doubles and uses all of the buddy block it gets; the elements in use are
// copied and the old array is freed, so every pointer into the vector dies.
// (A later change can reserve address space here instead, so that the
// arrays never move on hosts that have it.)
static void vec_reserve(IR_Vec *v, size_t need) {
    if (need <= v->cap) return;
    size_t want = v->cap ? (size_t)v->cap * 2 : 16;
    while (want < need) want *= 2;
    size_t actual = 0;
    void *p = buddy_alloc(want * v->elem, &actual);
    if (!p) {
        fprintf(stderr, "MLIR: out of memory growing an IR vector to %zu elements\n", want);
        abort();
    }
    if (v->n) memcpy(p, v->data, (size_t)v->n * v->elem);
    if (v->data) buddy_free(v->data);
    v->data = p;
    size_t cap = actual / v->elem;
    v->cap = cap > 0xffffffffu ? 0xffffffffu : (uint32_t)cap;
    v->gen++;
}

static void vec_init(IR_Vec *v, uint32_t elem) {
    v->data = NULL;
    v->n = 0;
    v->cap = 0;
    v->elem = elem;
    v->gen = 0;
    vec_reserve(v, 16);
    memset(v->data, 0, elem);  // element 0: the invalid handle's record
    v->n = 1;
}

static void vec_free(IR_Vec *v) {
    if (v->data) buddy_free(v->data);
    v->data = NULL;
    v->n = v->cap = 0;
}

// Appends `k` uninitialized elements and returns the index of the first.
static inline uint32_t vec_push(IR_Vec *v, uint32_t k) {
    if ((size_t)v->n + k > v->cap) vec_reserve(v, (size_t)v->n + k);
    uint32_t i = v->n;
    v->n += k;
    return i;
}

// Pointer-holding sections check that the vectors they point into did not
// move (the rule above).
#ifdef MLIR_API_CHECKS
#define GEN_SAVE(c, vec) uint32_t gen_##vec = (c)->vec.gen
#define GEN_CHECK(c, vec, fn) API_CHECK((c)->vec.gen == gen_##vec, fn, \
        "internal: a pointer into the " #vec " vector was held across its growth")
#else
#define GEN_SAVE(c, vec) ((void)0)
#define GEN_CHECK(c, vec, fn) ((void)0)
#endif

// Handles. Natively a handle is the element's index. Checked builds tag it
// with its kind in the top four bits, so that passing a handle of another
// kind (or one from another context, when out of range) is caught; release
// builds use the bare index.
#ifdef MLIR_API_CHECKS
#define HK_OP 1u
#define HK_VALUE 2u
#define HK_BLOCK 3u
#define HK_REGION 4u
#define H_SHIFT 28
#define H_MASK 0x0fffffffu
static uint32_t h_index(uintptr_t h, uint32_t kind, uint32_t n, const char *fn) {
    if (h == 0) return 0;
    if ((uint32_t)(h >> H_SHIFT) != kind) api_check_failed(fn, "handle of another kind, or not a handle");
    uint32_t i = (uint32_t)(h & H_MASK);
    if (i >= n) api_check_failed(fn, "handle out of range (from another context?)");
    return i;
}
static inline uintptr_t h_make(uint32_t i, uint32_t kind) {
    if (i == 0) return 0;
    if (i > H_MASK) api_check_failed("MLIR", "too many IR objects for a checked build");
    return (uintptr_t)i | ((uintptr_t)kind << H_SHIFT);
}
#define OP_IDX(c, h, fn)     h_index((h), HK_OP, (c)->ops.n, (fn))
#define VAL_IDX(c, h, fn)    h_index((h), HK_VALUE, (c)->vals.n, (fn))
#define BLOCK_IDX(c, h, fn)  h_index((h), HK_BLOCK, (c)->blocks.n, (fn))
#define REGION_IDX(c, h, fn) h_index((h), HK_REGION, (c)->regions.n, (fn))
#define OP_H(i)     ((MLIR_OpHandle)h_make((i), HK_OP))
#define VAL_H(i)    ((MLIR_ValueHandle)h_make((i), HK_VALUE))
#define BLOCK_H(i)  ((MLIR_BlockHandle)h_make((i), HK_BLOCK))
#define REGION_H(i) ((MLIR_RegionHandle)h_make((i), HK_REGION))
#else
#define OP_IDX(c, h, fn)     ((uint32_t)(h))
#define VAL_IDX(c, h, fn)    ((uint32_t)(h))
#define BLOCK_IDX(c, h, fn)  ((uint32_t)(h))
#define REGION_IDX(c, h, fn) ((uint32_t)(h))
#define OP_H(i)     ((MLIR_OpHandle)(i))
#define VAL_H(i)    ((MLIR_ValueHandle)(i))
#define BLOCK_H(i)  ((MLIR_BlockHandle)(i))
#define REGION_H(i) ((MLIR_RegionHandle)(i))
#endif

// Under MLIR_API_CHECKS an erased op keeps this type, so every later use of
// its handle is caught. Without checks MLIR_EraseOp leaves the type alone.
#define OP_TYPE_ERASED_SENTINEL ((MLIR_OpType)0x7fffffff)

// Checks that op index `o` is a live op (not invalid, not erased). The
// accessors do not handle invalid handles in release builds: the API does
// not promise it (mlir_api.h, "Diagnostics"), and the zeroed element 0 makes
// most such reads harmless anyway.
#ifdef MLIR_API_CHECKS
static void check_live_op(MLIR_Context *c, uint32_t o, const char *fn) {
    API_CHECK(o != 0, fn, "op handle is NULL");
    API_CHECK(OPS(c)[o].op_type != OP_TYPE_ERASED_SENTINEL, fn, "op was erased");
}
#define CHECK_OP(c, o, fn) check_live_op((c), (o), (fn))
#else
#define CHECK_OP(c, o, fn) ((void)0)
#endif
// For functions that accept MLIR_INVALID_HANDLE (as upstream does) but not an
// erased op.
#define CHECK_NOT_ERASED(c, o, fn) \
    API_CHECK((o) == 0 || OPS(c)[o].op_type != OP_TYPE_ERASED_SENTINEL, fn, "op was erased")
#define CHECK_INDEX(i, n, fn) API_CHECK((size_t)(i) < (size_t)(n), fn, "index out of range")
// A typed attribute getter applied to the wrong kind.
#define CHECK_ATTR_KIND(ah, ok, fn, what) do { \
        API_CHECK((ah) != MLIR_INVALID_HANDLE, fn, "attribute handle is NULL"); \
        API_CHECK(ok, fn, what); \
    } while (0)

static inline IR_Type *resolve_type(MLIR_TypeHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Type *)(uintptr_t)h;
}

static inline IR_Attribute *resolve_attr(MLIR_AttributeHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Attribute *)(uintptr_t)h;
}

static inline IR_Location *resolve_loc(MLIR_LocationHandle h) {
    return h == MLIR_INVALID_HANDLE ? NULL : (IR_Location *)(uintptr_t)h;
}

// --- Def-use lists -----------------------------------------------------------
//
// Each value's uses are a doubly-linked list of operand records, newest
// first. MLIR_GetValueNumUses is O(1), MLIR_ReplaceAllUsesOfValue
// O(uses), MLIR_GetValueUseOwner O(idx). An operand is linked iff its op was
// created with tracking on (OPF_TRACKED) and its value has a use list
// (VAL_TRACKED); neither flag changes after creation.

static inline bool operand_linked(MLIR_Context *c, uint32_t owner, uint32_t value) {
    return value != 0 && (OPS(c)[owner].flags & OPF_TRACKED) &&
           (VALS(c)[value].index & VAL_TRACKED);
}

// Links operand `u` (its value already set) at the front of its value's list.
static inline void use_push(MLIR_Context *c, uint32_t u) {
    IR_Operand *rec = &OPNDS(c)[u];
    IR_Value *v = &VALS(c)[rec->value];
    rec->prev = 0;
    rec->next = v->first_use;
    if (v->first_use) OPNDS(c)[v->first_use].prev = u;
    v->first_use = u;
    v->n_uses++;
}

static inline void use_unlink(MLIR_Context *c, uint32_t u) {
    IR_Operand *rec = &OPNDS(c)[u];
    IR_Value *v = &VALS(c)[rec->value];
    if (rec->prev) OPNDS(c)[rec->prev].next = rec->next;
    else           v->first_use = rec->next;
    if (rec->next) OPNDS(c)[rec->next].prev = rec->prev;
    rec->next = rec->prev = 0;
    if (v->n_uses) v->n_uses--;
}

// Sets operand `u` of op `owner` to `value`, keeping the use lists right.
static void operand_set(MLIR_Context *c, uint32_t owner, uint32_t u, uint32_t value) {
    uint32_t old = OPNDS(c)[u].value;
    if (old == value) return;
    if (operand_linked(c, owner, old)) use_unlink(c, u);
    OPNDS(c)[u].value = value;
    if (operand_linked(c, owner, value)) use_push(c, u);
}

// Moves operand record `from` to `to` (an unused record), keeping its place
// in its value's use list.
static void operand_move(MLIR_Context *c, uint32_t from, uint32_t to) {
    IR_Operand *o = OPNDS(c);
    o[to] = o[from];
    if (operand_linked(c, o[to].owner, o[to].value)) {
        if (o[to].prev) o[o[to].prev].next = to;
        else            VALS(c)[o[to].value].first_use = to;
        if (o[to].next) o[o[to].next].prev = to;
    }
    o[from].value = 0;
    o[from].next = o[from].prev = 0;
}

// A new value record. A value has a use list if the context tracks uses
// when it is created, or if it has a register name (as before #354).
static uint32_t value_create(MLIR_Context *c, uint32_t def, uint32_t index,
                             MLIR_TypeHandle type, MLIR_LocationHandle loc,
                             string register_name) {
    uint32_t v = vec_push(&c->vals, 1);
    IR_Value *p = &VALS(c)[v];
    p->type = type;
    p->location = loc;
    p->def = def;
    p->index = index & VAL_INDEX_MASK;
    if (!c->no_def_use_tracking || register_name.size != 0) p->index |= VAL_TRACKED;
    p->first_use = 0;
    p->n_uses = 0;
    p->name = 0;
    p->pad = 0;
    if (register_name.size != 0) {
        uint32_t n = vec_push(&c->names, 1);
        NAMES(c)[n] = register_name;
        VALS(c)[v].name = n;
    }
    return v;
}

// The op's aux record, created on first use.
static uint32_t op_aux_get(MLIR_Context *c, uint32_t o) {
    uint32_t a = OPS(c)[o].aux;
    if (a) return a;
    a = vec_push(&c->op_aux, 1);
    IR_OpAux *p = &OPAUX(c)[a];
    p->unnumbered_loc_def = MLIR_INVALID_HANDLE;
    p->trailing_comment = (string){0};
    p->source_line_start = -1;
    p->first_result_type = 0;
    p->n_result_types = 0;
    p->explicit_result_types = false;
    OPS(c)[o].aux = a;
    return a;
}

// Grows an arena array of 32-bit indices so it holds `need` entries; the
// old array is left to the arena.
static uint32_t *u32_reserve(Arena *arena, uint32_t *old, uint32_t n, uint32_t *cap, size_t need) {
    if (need <= *cap) return old;
    size_t nc = *cap ? (size_t)*cap * 2 : 4;
    while (nc < need) nc *= 2;
    uint32_t *p = arena_new_array(arena, uint32_t, nc);
    if (n) memcpy(p, old, (size_t)n * sizeof(uint32_t));
    *cap = (uint32_t)nc;
    return p;
}

// Removes every occurrence of `x` from `arr[0..*n)`, keeping the order.
static void u32_remove(uint32_t *arr, uint32_t *n, uint32_t x) {
    uint32_t w = 0;
    for (uint32_t i = 0; i < *n; i++)
        if (arr[i] != x) arr[w++] = arr[i];
    *n = w;
}

// Compare two IR_Type structures for value-equality. Used to intern
// types so that two semantically-equivalent types share the same
// MLIR_TypeHandle (matching upstream MLIR's IntegerType::get etc.).
// This makes `t1 == t2` a valid type-identity check across the API —
// e.g., the get_undef_value / get_switch_value caches in the
// cf->scf lifter rely on it.
static bool type_eq(const IR_Type *a, const IR_Type *b) {
    if (a->kind != b->kind) return false;
    switch (a->kind) {
        case TYPE_KIND_UNKNOWN:
        case TYPE_KIND_OPAQUE:
        case TYPE_KIND_INDEX:
        case TYPE_KIND_LLVM_VOID:
            return true;
        case TYPE_KIND_INTEGER:
            return a->data.integer.width == b->data.integer.width &&
                   a->data.integer.is_signed == b->data.integer.is_signed;
        case TYPE_KIND_FLOAT:
            return a->data.floating.width == b->data.floating.width &&
                   a->data.floating.is_bfloat == b->data.floating.is_bfloat;
        case TYPE_KIND_TENSOR:
        case TYPE_KIND_MEMREF:
            if (a->data.shaped.element_type != b->data.shaped.element_type) return false;
            if (a->data.shaped.rank != b->data.shaped.rank) return false;
            if (a->data.shaped.rank == 0) return true;
            if (!a->data.shaped.shape || !b->data.shaped.shape)
                return a->data.shaped.shape == b->data.shaped.shape;
            return memcmp(a->data.shaped.shape, b->data.shaped.shape,
                          (size_t)a->data.shaped.rank * sizeof(int64_t)) == 0;
        case TYPE_KIND_POINTER:
            return a->data.pointer.element_type == b->data.pointer.element_type &&
                   a->data.pointer.has_address_space == b->data.pointer.has_address_space &&
                   a->data.pointer.address_space == b->data.pointer.address_space;
        case TYPE_KIND_LLVM_PTR:
            return a->data.llvm_pointer.address_space ==
                   b->data.llvm_pointer.address_space;
        case TYPE_KIND_LLVM_ARRAY:
            return a->data.llvm_array.element == b->data.llvm_array.element &&
                   a->data.llvm_array.count == b->data.llvm_array.count;
        case TYPE_KIND_LLVM_FUNCTION:
            if (a->data.llvm_function.result != b->data.llvm_function.result) return false;
            if (a->data.llvm_function.is_var_arg != b->data.llvm_function.is_var_arg) return false;
            if (a->data.llvm_function.n_inputs != b->data.llvm_function.n_inputs) return false;
            if (a->data.llvm_function.n_inputs == 0) return true;
            return memcmp(a->data.llvm_function.inputs, b->data.llvm_function.inputs,
                          a->data.llvm_function.n_inputs * sizeof(MLIR_TypeHandle)) == 0;
        case TYPE_KIND_FUNCTION:
            if (a->data.function.n_inputs != b->data.function.n_inputs) return false;
            if (a->data.function.n_results != b->data.function.n_results) return false;
            if (a->data.function.n_inputs > 0 &&
                memcmp(a->data.function.inputs, b->data.function.inputs,
                       a->data.function.n_inputs * sizeof(MLIR_TypeHandle)) != 0) return false;
            if (a->data.function.n_results > 0 &&
                memcmp(a->data.function.results, b->data.function.results,
                       a->data.function.n_results * sizeof(MLIR_TypeHandle)) != 0) return false;
            return true;
        case TYPE_KIND_LLVM_STRUCT:
            // Identified (name.size > 0) handled by intern_llvm_struct.
            // We don't intern LLVM struct types through this path; the
            // dedicated intern_llvm_struct keeps name->handle identity.
            return false;
    }
    return false;
}

// Copy the arrays a type points to into the context's persistent arena.
// Type constructors build the candidate IR_Type with pointers to the
// caller's (possibly stack) arrays, and only a miss in intern_type copies
// them, so creating an existing type allocates nothing.
static void type_copy_payload(Arena *a, IR_Type *t) {
    switch (t->kind) {
        case TYPE_KIND_TENSOR:
        case TYPE_KIND_MEMREF:
            if (t->data.shaped.rank > 0 && t->data.shaped.shape) {
                int64_t *shape = arena_new_array(a, int64_t, t->data.shaped.rank);
                memcpy(shape, t->data.shaped.shape,
                       (size_t)t->data.shaped.rank * sizeof(int64_t));
                t->data.shaped.shape = shape;
            }
            break;
        case TYPE_KIND_FUNCTION:
            if (t->data.function.n_inputs > 0) {
                MLIR_TypeHandle *in = arena_new_array(a, MLIR_TypeHandle, t->data.function.n_inputs);
                memcpy(in, t->data.function.inputs,
                       t->data.function.n_inputs * sizeof(MLIR_TypeHandle));
                t->data.function.inputs = in;
            }
            if (t->data.function.n_results > 0) {
                MLIR_TypeHandle *out = arena_new_array(a, MLIR_TypeHandle, t->data.function.n_results);
                memcpy(out, t->data.function.results,
                       t->data.function.n_results * sizeof(MLIR_TypeHandle));
                t->data.function.results = out;
            }
            break;
        case TYPE_KIND_LLVM_FUNCTION:
            if (t->data.llvm_function.n_inputs > 0) {
                MLIR_TypeHandle *in = arena_new_array(a, MLIR_TypeHandle, t->data.llvm_function.n_inputs);
                memcpy(in, t->data.llvm_function.inputs,
                       t->data.llvm_function.n_inputs * sizeof(MLIR_TypeHandle));
                t->data.llvm_function.inputs = in;
            }
            break;
        default:
            break;
    }
}

// Types are uniqued per context, so two structurally equal types share one
// handle (as upstream's IntegerType::get etc.): `t1 == t2` is a valid
// type-identity check across the API, e.g. for the get_undef_value /
// get_switch_value caches in the cf->scf lifter. The type, its arrays and
// the table live in the persistent arena, so they survive the IR arena.
static MLIR_TypeHandle intern_type(MLIR_Context *ctx, IR_Type t) {
    if (!ctx) return MLIR_INVALID_HANDLE;
    for (size_t i = 0; i < ctx->n_types; i++) {
        IR_Type *cur = resolve_type(ctx->type_handles[i]);
        if (cur && type_eq(cur, &t)) return ctx->type_handles[i];
    }
    Arena *a = ctx->persistent;
    type_copy_payload(a, &t);
    IR_Type *slot = arena_new(a, IR_Type);
    *slot = t;
    MLIR_TypeHandle h = (MLIR_TypeHandle)slot;
    if (ctx->n_types == ctx->cap_types) {
        size_t nc = ctx->cap_types ? ctx->cap_types * 2 : 64;
        MLIR_TypeHandle *nh = arena_new_array(a, MLIR_TypeHandle, nc);
        if (ctx->n_types) {
            memcpy(nh, ctx->type_handles, ctx->n_types * sizeof(MLIR_TypeHandle));
        }
        ctx->type_handles = nh;
        ctx->cap_types = nc;
    }
    ctx->type_handles[ctx->n_types++] = h;
    return h;
}

static inline MLIR_AttributeHandle alloc_attr_obj(MLIR_Context *ctx, IR_Attribute a) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Attribute *slot = arena_new(ctx->arena, IR_Attribute);
    *slot = a;
    return (MLIR_AttributeHandle)(uintptr_t)slot;
}

// Copy `s` into ctx->arena so the returned string is owned by the context's
// arena rather than aliasing caller-provided memory. Every attribute
// constructor copies its name (and string payload) this way, as upstream does. Multi-pass pipelines
// (wasm->macho) free intermediate arenas between passes, so any string stored
// in an IR object that must outlive its producing pass has to live in the
// destination arena. Empty/NULL strings are returned unchanged.
static inline string arena_dup_str(MLIR_Context *ctx, string s) {
    if (!ctx || !ctx->arena || s.size == 0 || !s.str) return s;
    char *p = arena_new_array(ctx->arena, char, s.size);
    memcpy(p, s.str, s.size);
    return (string){ p, s.size };
}

static inline MLIR_LocationHandle alloc_loc(MLIR_Context *ctx, IR_Location l) {
    if (!ctx || !ctx->arena) return MLIR_INVALID_HANDLE;
    IR_Location *slot = arena_new(ctx->arena, IR_Location);
    *slot = l;
    return (MLIR_LocationHandle)(uintptr_t)slot;
}

// Context
//
// The context lives in its own persistent arena, together with the types and
// the interning tables, so destroying it is one arena_destroy plus the IR
// vectors. 64 KiB minus the chunk header keeps the first chunk within one
// 64 KiB buddy block.
MLIR_Context *MLIR_CreateContext(const MLIR_ContextOptions *options) {
    Arena *persistent = arena_create(64 * 1024 - 256);
    if (!persistent) return NULL;
    MLIR_Context *ctx = arena_new(persistent, MLIR_Context);
    *ctx = (MLIR_Context){0};
    ctx->persistent = persistent;
    op_type_table_init(&ctx->op_types, persistent);
    vec_init(&ctx->ops, (uint32_t)sizeof(IR_Op));
    vec_init(&ctx->vals, (uint32_t)sizeof(IR_Value));
    vec_init(&ctx->opnds, (uint32_t)sizeof(IR_Operand));
    vec_init(&ctx->extra, (uint32_t)sizeof(uint32_t));
    vec_init(&ctx->ptrs, (uint32_t)sizeof(uintptr_t));
    vec_init(&ctx->blocks, (uint32_t)sizeof(IR_Block));
    vec_init(&ctx->regions, (uint32_t)sizeof(IR_Region));
    vec_init(&ctx->op_aux, (uint32_t)sizeof(IR_OpAux));
    vec_init(&ctx->names, (uint32_t)sizeof(string));
    if (options) {
        ctx->arena = options->arena;
        ctx->no_def_use_tracking = options->no_def_use_tracking;
        if (options->expected_ops) MLIR_SetContextExpectedOps(ctx, options->expected_ops);
    }
    return ctx;
}

// Reserves the IR vectors for `n` more ops, in proportions that cover
// tinyc's pipelines (per op, the C pipeline makes about 0.8 values, 1.1
// operands, 0.2 region/successor words, 0.75 attributes, 0.05 blocks and
// 0.02 regions; the wasm lift 0.5, 0.7, 0.03, 1.4, 0.02 and 0.01).
// Reserving more than is used costs address space, not memory: the pages
// are not touched.
void MLIR_SetContextExpectedOps(MLIR_Context *ctx, size_t n) {
    API_CHECK(ctx != NULL, "MLIR_SetContextExpectedOps", "context is NULL");
    if (n > 0x0fffffff) n = 0x0fffffff;
    vec_reserve(&ctx->ops, ctx->ops.n + n);
    vec_reserve(&ctx->vals, ctx->vals.n + n);
    vec_reserve(&ctx->opnds, ctx->opnds.n + n + n / 4);
    vec_reserve(&ctx->extra, ctx->extra.n + n / 4);
    vec_reserve(&ctx->ptrs, ctx->ptrs.n + n + n / 2);
    vec_reserve(&ctx->blocks, ctx->blocks.n + n / 16);
    vec_reserve(&ctx->regions, ctx->regions.n + n / 32);
}

void MLIR_DestroyContext(MLIR_Context *ctx) {
    if (!ctx) return;
    vec_free(&ctx->ops);
    vec_free(&ctx->vals);
    vec_free(&ctx->opnds);
    vec_free(&ctx->extra);
    vec_free(&ctx->ptrs);
    vec_free(&ctx->blocks);
    vec_free(&ctx->regions);
    vec_free(&ctx->op_aux);
    vec_free(&ctx->names);
    arena_destroy(ctx->persistent);
}

void MLIR_SetArenaAllocator(MLIR_Context *ctx, Arena *arena) {
    API_CHECK(ctx != NULL, "MLIR_SetArenaAllocator", "context is NULL");
    if (!ctx) return;
    ctx->arena = arena;
}

Arena *MLIR_GetArenaAllocator(MLIR_Context *ctx) {
    API_CHECK(ctx != NULL, "MLIR_GetArenaAllocator", "context is NULL");
    return ctx ? ctx->arena : NULL;
}

void MLIR_SetContextDefUseTracking(MLIR_Context *ctx, bool enabled) {
    API_CHECK(ctx != NULL, "MLIR_SetContextDefUseTracking", "context is NULL");
    ctx->no_def_use_tracking = !enabled;
}

bool MLIR_HasContextDefUseTracking(MLIR_Context *ctx) {
    API_CHECK(ctx != NULL, "MLIR_HasContextDefUseTracking", "context is NULL");
    return !ctx->no_def_use_tracking;
}

// Diagnostics

static void default_diagnostic_handler(void *user, MLIR_DiagSeverity severity,
                                       MLIR_LocationHandle loc, string message) {
    (void)user;
    const char *sev = "error";
    if (severity == MLIR_DIAG_WARNING) sev = "warning";
    else if (severity == MLIR_DIAG_NOTE) sev = "note";
    else if (severity == MLIR_DIAG_REMARK) sev = "remark";
    if (loc != MLIR_INVALID_HANDLE && MLIR_GetLocationKind(loc) == MLIR_LOC_FILE) {
        string file = MLIR_GetLocationFileFilename(loc);
        fprintf(stderr, "%.*s:%d:%d: %s: %.*s\n", (int)file.size, file.str,
                MLIR_GetLocationFileLine(loc), MLIR_GetLocationFileColumn(loc),
                sev, (int)message.size, message.str);
    } else if (loc != MLIR_INVALID_HANDLE && MLIR_GetLocationKind(loc) != MLIR_LOC_UNKNOWN &&
               ((IR_Location *)loc)->original_text.size > 0) {
        // Other locations print in their `loc(...)` form, as upstream's do.
        string text = ((IR_Location *)loc)->original_text;
        fprintf(stderr, "%.*s: %s: %.*s\n", (int)text.size, text.str, sev,
                (int)message.size, message.str);
    } else {
        fprintf(stderr, "%s: %.*s\n", sev, (int)message.size, message.str);
    }
}

void MLIR_SetDiagnosticHandler(MLIR_Context *ctx, MLIR_DiagnosticHandler fn, void *user) {
    API_CHECK(ctx != NULL, "MLIR_SetDiagnosticHandler", "context is NULL");
    ctx->diag_fn = fn;
    ctx->diag_user = fn ? user : NULL;
}

// Short messages are formatted on the stack, longer ones in a heap buffer
// that is freed after the handler returns (the message is borrowed).
void MLIR_EmitDiagnostic(MLIR_Context *ctx, MLIR_DiagSeverity severity,
                         MLIR_LocationHandle loc, const char *fmt, ...) {
    API_CHECK(ctx != NULL, "MLIR_EmitDiagnostic", "context is NULL");
    char local[256];
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(local, sizeof(local), fmt, ap);
    va_end(ap);
    if (n < 0) n = 0;
    char *buf = local;
    if ((size_t)n >= sizeof(local)) {
        buf = (char *)malloc((size_t)n + 1);
        if (buf) vsnprintf(buf, (size_t)n + 1, fmt, ap2);
        else { buf = local; n = (int)sizeof(local) - 1; }
    }
    va_end(ap2);
    string message;
    message.str = buf;
    message.size = (size_t)n;
    if (ctx->diag_fn)
        ctx->diag_fn(ctx->diag_user, severity, loc, message);
    else
        default_diagnostic_handler(NULL, severity, loc, message);
    if (buf != local) free(buf);
}

// Operation creation
//
// The results must be consecutive values (created one after another by
// MLIR_CreateValueOpResult, or the results of an op this one replaces):
// result i is then first_result + i. Checked with MLIR_API_CHECKS.
MLIR_OpHandle MLIR_CreateOpWithSuccessors(
    MLIR_Context *ctx,
    MLIR_OpType type,
    MLIR_AttributeHandle *attributes, size_t n_attributes,
    MLIR_TypeHandle *result_types, size_t n_result_types,
    MLIR_ValueHandle *results, size_t n_results,
    MLIR_ValueHandle *operands, size_t n_operands,
    MLIR_RegionHandle *regions, size_t n_regions,
    MLIR_BlockHandle *successors, size_t n_successors,
    MLIR_ValueHandle **successor_operands,
    size_t *n_successor_operands,
    MLIR_LocationHandle location,
    MLIR_LocationHandle unnumbered_loc_def,
    string trailing_comment,
    int64_t source_line_start) {

    API_CHECK(ctx != NULL, "MLIR_CreateOp", "context is NULL");
    API_CHECK(type != OP_TYPE_UNREGISTERED && (uint32_t)type < ctx->op_types.n_types,
              "MLIR_CreateOp", "not an op type of this context");
    API_CHECK(n_attributes <= 0xffff && n_regions <= 0xffff && n_successors <= 0xffff,
              "MLIR_CreateOp", "too many attributes, regions or successors");

    size_t n_succ_ops = 0;
    if (n_successor_operands)
        for (size_t s = 0; s < n_successors; s++) n_succ_ops += n_successor_operands[s];
    size_t n_all = n_operands + n_succ_ops;
    size_t n_extra = n_regions + 2 * n_successors;

    uint32_t first_result = 0;
    if (n_results > 0) {
        first_result = VAL_IDX(ctx, results[0], "MLIR_CreateOp");
#ifdef MLIR_API_CHECKS
        for (size_t i = 0; i < n_results; i++)
            API_CHECK(first_result != 0 &&
                      VAL_IDX(ctx, results[i], "MLIR_CreateOp") == first_result + i,
                      "MLIR_CreateOp", "the results are not consecutive values");
#endif
    }

    // Allocate everything first: growing a vector moves it.
    uint32_t op = vec_push(&ctx->ops, 1);
    uint32_t first_operand = vec_push(&ctx->opnds, (uint32_t)n_all);
    uint32_t first_extra = vec_push(&ctx->extra, (uint32_t)n_extra);
    uint32_t first_attr = vec_push(&ctx->ptrs, (uint32_t)n_attributes);
    // From here to the aux record, pointers into these vectors are held.
    GEN_SAVE(ctx, ops);
    GEN_SAVE(ctx, opnds);
    GEN_SAVE(ctx, extra);
    GEN_SAVE(ctx, ptrs);
    GEN_SAVE(ctx, vals);

    bool track = !ctx->no_def_use_tracking;
    IR_Op *o = &OPS(ctx)[op];
    o->op_type = type;
    o->parent_block = 0;
    o->first_operand = first_operand;
    o->n_operands = (uint32_t)n_operands;
    o->n_succ_operands = (uint32_t)n_succ_ops;
    o->first_result = first_result;
    o->n_results = (uint32_t)n_results;
    o->first_extra = first_extra;
    o->n_regions = (uint16_t)n_regions;
    o->n_successors = (uint16_t)n_successors;
    o->first_attr = first_attr;
    o->n_attributes = (uint16_t)n_attributes;
    o->flags = track ? OPF_TRACKED : 0;
    o->aux = 0;
    o->location = location;

    // Operands: the regular ones, then each successor's, linked in that
    // order (each at the front of its value's use list).
    IR_Operand *rec = OPNDS(ctx);
    uint32_t u = first_operand;
    for (size_t i = 0; i < n_operands; i++, u++) {
        rec[u].value = VAL_IDX(ctx, operands[i], "MLIR_CreateOp");
        rec[u].owner = op;
        rec[u].next = rec[u].prev = 0;
        if (operand_linked(ctx, op, rec[u].value)) use_push(ctx, u);
    }
    uint32_t *ex = EXTRA(ctx) + first_extra;
    for (size_t i = 0; i < n_regions; i++) {
        uint32_t r = REGION_IDX(ctx, regions[i], "MLIR_CreateOp");
        ex[i] = r;
        // The regions now belong to this op (MLIR_GetBlockParentOp).
        if (r) REGIONS(ctx)[r].parent_op = op;
    }
    ex += n_regions;
    for (size_t s = 0; s < n_successors; s++) {
        size_t cnt = n_successor_operands ? n_successor_operands[s] : 0;
        ex[2 * s] = BLOCK_IDX(ctx, successors[s], "MLIR_CreateOp");
        ex[2 * s + 1] = (uint32_t)cnt;
        MLIR_ValueHandle *vs = successor_operands ? successor_operands[s] : NULL;
        for (size_t k = 0; k < cnt; k++, u++) {
            rec[u].value = vs ? VAL_IDX(ctx, vs[k], "MLIR_CreateOp") : 0;
            rec[u].owner = op;
            rec[u].next = rec[u].prev = 0;
            if (operand_linked(ctx, op, rec[u].value)) use_push(ctx, u);
        }
    }
    uintptr_t *pa = PTRS(ctx) + first_attr;
    for (size_t i = 0; i < n_attributes; i++) pa[i] = (uintptr_t)attributes[i];

    // The results get this op as their definition, and a type from
    // result_types if they have none. The op's result types are its results'
    // types unless the two disagree; then they are stored (rare).
    bool explicit_types = n_result_types != n_results;
    IR_Value *res = VALS(ctx) + first_result;
    for (size_t i = 0; i < n_results; i++) {
        res[i].def = op;
        if (i < n_result_types) {
            if (res[i].type == MLIR_INVALID_HANDLE) res[i].type = result_types[i];
            else if (res[i].type != result_types[i]) explicit_types = true;
        }
    }
    GEN_CHECK(ctx, ops, "MLIR_CreateOp");
    GEN_CHECK(ctx, opnds, "MLIR_CreateOp");
    GEN_CHECK(ctx, extra, "MLIR_CreateOp");
    GEN_CHECK(ctx, ptrs, "MLIR_CreateOp");
    GEN_CHECK(ctx, vals, "MLIR_CreateOp");

    if (unnumbered_loc_def != MLIR_INVALID_HANDLE || trailing_comment.size != 0 ||
        source_line_start != -1 || explicit_types) {
        uint32_t a = op_aux_get(ctx, op);
        OPAUX(ctx)[a].unnumbered_loc_def = unnumbered_loc_def;
        OPAUX(ctx)[a].trailing_comment = trailing_comment;
        OPAUX(ctx)[a].source_line_start = (int32_t)source_line_start;
        if (explicit_types) {
            uint32_t ft = vec_push(&ctx->ptrs, (uint32_t)n_result_types);
            for (size_t i = 0; i < n_result_types; i++)
                PTRS(ctx)[ft + i] = (uintptr_t)result_types[i];
            OPAUX(ctx)[a].first_result_type = ft;
            OPAUX(ctx)[a].n_result_types = (uint32_t)n_result_types;
            OPAUX(ctx)[a].explicit_result_types = true;
        }
    }
    return OP_H(op);
}

MLIR_OpHandle MLIR_CreateOp(
    MLIR_Context *ctx,
    MLIR_OpType type,
    MLIR_AttributeHandle *attributes, size_t n_attributes,
    MLIR_TypeHandle *result_types, size_t n_result_types,
    MLIR_ValueHandle *results, size_t n_results,
    MLIR_ValueHandle *operands, size_t n_operands,
    MLIR_RegionHandle *regions, size_t n_regions,
    MLIR_LocationHandle location,
    MLIR_LocationHandle unnumbered_loc_def,
    string trailing_comment,
    int64_t source_line_start) {
    return MLIR_CreateOpWithSuccessors(
        ctx, type,
        attributes, n_attributes,
        result_types, n_result_types,
        results, n_results,
        operands, n_operands,
        regions, n_regions,
        NULL, 0, NULL, NULL,
        location, unnumbered_loc_def, trailing_comment, source_line_start);
}

// Inserts op `o` at position `idx` (clamped) of block `b`.
static void block_insert_op(MLIR_Context *ctx, uint32_t b, uint32_t o, size_t idx) {
    IR_Block *blk = &BLOCKS(ctx)[b];
    if (idx > blk->n_ops) idx = blk->n_ops;
    blk->ops = u32_reserve(ctx->arena, blk->ops, blk->n_ops, &blk->cap_ops, (size_t)blk->n_ops + 1);
    if (idx < blk->n_ops)
        memmove(blk->ops + idx + 1, blk->ops + idx, (blk->n_ops - idx) * sizeof(uint32_t));
    blk->ops[idx] = o;
    blk->n_ops++;
    OPS(ctx)[o].parent_block = b;
}

void MLIR_AppendBlockOp(MLIR_Context *ctx, MLIR_BlockHandle bh, MLIR_OpHandle oh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_AppendBlockOp");
    uint32_t o = OP_IDX(ctx, oh, "MLIR_AppendBlockOp");
    API_CHECK(b != 0, "MLIR_AppendBlockOp", "block handle is NULL");
    CHECK_OP(ctx, o, "MLIR_AppendBlockOp");
    API_CHECK(OPS(ctx)[o].parent_block == 0, "MLIR_AppendBlockOp", "op is already in a block (use MLIR_MoveOpToBlockEnd)");
    block_insert_op(ctx, b, o, BLOCKS(ctx)[b].n_ops);
}

void MLIR_InsertBlockOpAtIndex(MLIR_Context *ctx, MLIR_BlockHandle bh, MLIR_OpHandle oh, size_t idx) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_InsertBlockOpAtIndex");
    uint32_t o = OP_IDX(ctx, oh, "MLIR_InsertBlockOpAtIndex");
    API_CHECK(b != 0, "MLIR_InsertBlockOpAtIndex", "block handle is NULL");
    CHECK_OP(ctx, o, "MLIR_InsertBlockOpAtIndex");
    API_CHECK(OPS(ctx)[o].parent_block == 0, "MLIR_InsertBlockOpAtIndex", "op is already in a block (detach it first)");
    block_insert_op(ctx, b, o, idx);
}

void MLIR_AppendBlockArg(MLIR_Context *ctx, MLIR_BlockHandle bh, MLIR_ValueHandle arg) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_AppendBlockArg");
    uint32_t v = VAL_IDX(ctx, arg, "MLIR_AppendBlockArg");
    API_CHECK(b != 0, "MLIR_AppendBlockArg", "block handle is NULL");
    IR_Block *blk = &BLOCKS(ctx)[b];
    blk->args = u32_reserve(ctx->arena, blk->args, blk->n_args, &blk->cap_args, (size_t)blk->n_args + 1);
    blk->args[blk->n_args++] = v;
    if (v) VALS(ctx)[v].def = b;
}

void MLIR_AppendRegionBlock(MLIR_Context *ctx, MLIR_RegionHandle rh, MLIR_BlockHandle bh) {
    uint32_t r = REGION_IDX(ctx, rh, "MLIR_AppendRegionBlock");
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_AppendRegionBlock");
    API_CHECK(r != 0 && b != 0, "MLIR_AppendRegionBlock", "region or block handle is NULL");
    API_CHECK(BLOCKS(ctx)[b].parent_region == 0, "MLIR_AppendRegionBlock",
              "block is already in a region (use MLIR_MoveBlockToRegionEnd)");
    IR_Region *reg = &REGIONS(ctx)[r];
    reg->blocks = u32_reserve(ctx->arena, reg->blocks, reg->n_blocks, &reg->cap_blocks,
                              (size_t)reg->n_blocks + 1);
    reg->blocks[reg->n_blocks++] = b;
    BLOCKS(ctx)[b].parent_region = r;
}

size_t MLIR_GetRegionNumBlocks(MLIR_Context *ctx, MLIR_RegionHandle rh) {
    uint32_t r = REGION_IDX(ctx, rh, "MLIR_GetRegionNumBlocks");
    API_CHECK(r != 0, "MLIR_GetRegionNumBlocks", "region handle is NULL");
    return REGIONS(ctx)[r].n_blocks;
}

MLIR_BlockHandle MLIR_GetRegionBlock(MLIR_Context *ctx, MLIR_RegionHandle rh, size_t idx) {
    uint32_t r = REGION_IDX(ctx, rh, "MLIR_GetRegionBlock");
    API_CHECK(r != 0, "MLIR_GetRegionBlock", "region handle is NULL");
    CHECK_INDEX(idx, REGIONS(ctx)[r].n_blocks, "MLIR_GetRegionBlock");
    return BLOCK_H(REGIONS(ctx)[r].blocks[idx]);
}

size_t MLIR_GetBlockNumOps(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockNumOps");
    API_CHECK(b != 0, "MLIR_GetBlockNumOps", "block handle is NULL");
    return BLOCKS(ctx)[b].n_ops;
}

MLIR_OpHandle MLIR_GetBlockOp(MLIR_Context *ctx, MLIR_BlockHandle bh, size_t idx) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockOp");
    API_CHECK(b != 0, "MLIR_GetBlockOp", "block handle is NULL");
    CHECK_INDEX(idx, BLOCKS(ctx)[b].n_ops, "MLIR_GetBlockOp");
    return OP_H(BLOCKS(ctx)[b].ops[idx]);
}

MLIR_OpType MLIR_GetOpType(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpType");
    CHECK_OP(ctx, o, "MLIR_GetOpType");
    return OPS(ctx)[o].op_type;
}

size_t MLIR_GetOpNumRegions(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpNumRegions");
    CHECK_OP(ctx, o, "MLIR_GetOpNumRegions");
    return OPS(ctx)[o].n_regions;
}

MLIR_RegionHandle MLIR_GetOpRegion(MLIR_Context *ctx, MLIR_OpHandle oh, size_t idx) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpRegion");
    CHECK_OP(ctx, o, "MLIR_GetOpRegion");
    CHECK_INDEX(idx, OPS(ctx)[o].n_regions, "MLIR_GetOpRegion");
    return REGION_H(EXTRA(ctx)[OPS(ctx)[o].first_extra + idx]);
}

string MLIR_GetOpTrailingComment(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpTrailingComment");
    CHECK_OP(ctx, o, "MLIR_GetOpTrailingComment");
    uint32_t a = OPS(ctx)[o].aux;
    return a ? OPAUX(ctx)[a].trailing_comment : str_lit("");
}

size_t MLIR_GetOpNumAttributes(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpNumAttributes");
    CHECK_OP(ctx, o, "MLIR_GetOpNumAttributes");
    return OPS(ctx)[o].n_attributes;
}

MLIR_AttributeHandle MLIR_GetOpAttribute(MLIR_Context *ctx, MLIR_OpHandle oh, size_t idx) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpAttribute");
    CHECK_OP(ctx, o, "MLIR_GetOpAttribute");
    CHECK_INDEX(idx, OPS(ctx)[o].n_attributes, "MLIR_GetOpAttribute");
    return (MLIR_AttributeHandle)PTRS(ctx)[OPS(ctx)[o].first_attr + idx];
}

MLIR_AttributeHandle MLIR_GetOpAttributeByName(MLIR_Context *ctx, MLIR_OpHandle oh, const char *name) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpAttributeByName");
    CHECK_OP(ctx, o, "MLIR_GetOpAttributeByName");
    API_CHECK(name != NULL, "MLIR_GetOpAttributeByName", "name is NULL");
    IR_Op *op = &OPS(ctx)[o];
    const uintptr_t *attrs = PTRS(ctx) + op->first_attr;
    size_t nlen = strlen(name);
    for (size_t i = 0; i < op->n_attributes; i++) {
        string an = MLIR_GetAttributeName((MLIR_AttributeHandle)attrs[i]);
        if (an.size == nlen && memcmp(an.str, name, nlen) == 0)
            return (MLIR_AttributeHandle)attrs[i];
    }
    return MLIR_INVALID_HANDLE;
}

size_t MLIR_GetOpNumOperands(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpNumOperands");
    CHECK_OP(ctx, o, "MLIR_GetOpNumOperands");
    return OPS(ctx)[o].n_operands;
}

MLIR_ValueHandle MLIR_GetOpOperand(MLIR_Context *ctx, MLIR_OpHandle oh, size_t idx) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpOperand");
    CHECK_OP(ctx, o, "MLIR_GetOpOperand");
    CHECK_INDEX(idx, OPS(ctx)[o].n_operands, "MLIR_GetOpOperand");
    return VAL_H(OPNDS(ctx)[OPS(ctx)[o].first_operand + idx].value);
}

size_t MLIR_GetOpNumResults(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpNumResults");
    CHECK_OP(ctx, o, "MLIR_GetOpNumResults");
    return OPS(ctx)[o].n_results;
}

MLIR_ValueHandle MLIR_GetOpResult(MLIR_Context *ctx, MLIR_OpHandle oh, size_t idx) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpResult");
    CHECK_OP(ctx, o, "MLIR_GetOpResult");
    CHECK_INDEX(idx, OPS(ctx)[o].n_results, "MLIR_GetOpResult");
    return VAL_H(OPS(ctx)[o].first_result + (uint32_t)idx);
}

MLIR_LocationHandle MLIR_GetOpLocation(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpLocation");
    CHECK_OP(ctx, o, "MLIR_GetOpLocation");
    return OPS(ctx)[o].location;
}

size_t MLIR_GetOpNumResultTypes(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpNumResultTypes");
    CHECK_OP(ctx, o, "MLIR_GetOpNumResultTypes");
    uint32_t a = OPS(ctx)[o].aux;
    if (a && OPAUX(ctx)[a].explicit_result_types) return OPAUX(ctx)[a].n_result_types;
    return OPS(ctx)[o].n_results;
}

MLIR_TypeHandle MLIR_GetOpResultType(MLIR_Context *ctx, MLIR_OpHandle oh, size_t idx) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpResultType");
    CHECK_OP(ctx, o, "MLIR_GetOpResultType");
    CHECK_INDEX(idx, MLIR_GetOpNumResultTypes(ctx, oh), "MLIR_GetOpResultType");
    uint32_t a = OPS(ctx)[o].aux;
    if (a && OPAUX(ctx)[a].explicit_result_types)
        return (MLIR_TypeHandle)PTRS(ctx)[OPAUX(ctx)[a].first_result_type + idx];
    return VALS(ctx)[OPS(ctx)[o].first_result + idx].type;
}

size_t MLIR_GetOpNumSuccessors(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpNumSuccessors");
    CHECK_OP(ctx, o, "MLIR_GetOpNumSuccessors");
    return OPS(ctx)[o].n_successors;
}

// The (block, operand count) pair of successor `s` of op `o`.
static inline uint32_t *op_succ_pair(MLIR_Context *ctx, uint32_t o, size_t s) {
    IR_Op *op = &OPS(ctx)[o];
    return EXTRA(ctx) + op->first_extra + op->n_regions + 2 * s;
}

// The flat operand position (from the op's first operand) of successor
// operand (s, 0).
static uint32_t op_succ_operand_base(MLIR_Context *ctx, uint32_t o, size_t s) {
    IR_Op *op = &OPS(ctx)[o];
    const uint32_t *pairs = EXTRA(ctx) + op->first_extra + op->n_regions;
    uint32_t off = op->n_operands;
    for (size_t k = 0; k < s; k++) off += pairs[2 * k + 1];
    return off;
}

MLIR_BlockHandle MLIR_GetOpSuccessor(MLIR_Context *ctx, MLIR_OpHandle oh, size_t idx) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpSuccessor");
    CHECK_OP(ctx, o, "MLIR_GetOpSuccessor");
    CHECK_INDEX(idx, OPS(ctx)[o].n_successors, "MLIR_GetOpSuccessor");
    return BLOCK_H(op_succ_pair(ctx, o, idx)[0]);
}

size_t MLIR_GetOpNumSuccessorOperands(MLIR_Context *ctx, MLIR_OpHandle oh, size_t s) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpNumSuccessorOperands");
    CHECK_OP(ctx, o, "MLIR_GetOpNumSuccessorOperands");
    CHECK_INDEX(s, OPS(ctx)[o].n_successors, "MLIR_GetOpNumSuccessorOperands");
    return op_succ_pair(ctx, o, s)[1];
}

MLIR_ValueHandle MLIR_GetOpSuccessorOperand(MLIR_Context *ctx, MLIR_OpHandle oh, size_t s, size_t i) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpSuccessorOperand");
    CHECK_OP(ctx, o, "MLIR_GetOpSuccessorOperand");
    CHECK_INDEX(s, OPS(ctx)[o].n_successors, "MLIR_GetOpSuccessorOperand");
    CHECK_INDEX(i, op_succ_pair(ctx, o, s)[1], "MLIR_GetOpSuccessorOperand");
    uint32_t u = OPS(ctx)[o].first_operand + op_succ_operand_base(ctx, o, s) + (uint32_t)i;
    return VAL_H(OPNDS(ctx)[u].value);
}

size_t MLIR_GetBlockNumArgs(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockNumArgs");
    API_CHECK(b != 0, "MLIR_GetBlockNumArgs", "block handle is NULL");
    return BLOCKS(ctx)[b].n_args;
}

MLIR_ValueHandle MLIR_GetBlockArg(MLIR_Context *ctx, MLIR_BlockHandle bh, size_t idx) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockArg");
    API_CHECK(b != 0, "MLIR_GetBlockArg", "block handle is NULL");
    CHECK_INDEX(idx, BLOCKS(ctx)[b].n_args, "MLIR_GetBlockArg");
    return VAL_H(BLOCKS(ctx)[b].args[idx]);
}

size_t MLIR_GetBlockIndex(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockIndex");
    if (!b) return (size_t)-1;
    uint32_t r = BLOCKS(ctx)[b].parent_region;
    if (!r) return (size_t)-1;
    IR_Region *reg = &REGIONS(ctx)[r];
    for (uint32_t i = 0; i < reg->n_blocks; i++)
        if (reg->blocks[i] == b) return i;
    return (size_t)-1;
}

MLIR_LocationHandle MLIR_GetValueLocation(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_GetValueLocation");
    return VALS(ctx)[v].location;
}

// Stubs for two mlir_api_upstream.h entry points, so that the `parser` test
// driver, which can select them at run time, links with this backend too.
string MLIR_PrintOperationUpstream(MLIR_Context *ctx, MLIR_OpHandle op) {
    (void)ctx; (void)op;
    return str_lit("error: MLIR_PrintOperationUpstream requires the upstream backend\n");
}

MLIR_OpHandle MLIR_ParseTextUpstream(MLIR_Context *ctx, string text) {
    (void)ctx; (void)text;
    return MLIR_INVALID_HANDLE;
}

// Type to string
string MLIR_GetTypeString(MLIR_Context *ctx, MLIR_TypeHandle th) {
    IR_Type *type = resolve_type(th);
    if (!type) return str_lit("null");
    if (!ctx || !ctx->arena) return str_lit("null");
    Arena *arena = ctx->arena;

    switch (type->kind) {
        case TYPE_KIND_UNKNOWN:
            return str_lit("?");
        case TYPE_KIND_OPAQUE:
            return str_lit("unknown");
        case TYPE_KIND_INTEGER:
            // Signless `i{w}` regardless of `is_signed` — MLIR integers are
            // signless by default and the upstream backend always produces
            // `i{w}` from `MLIR_CreateTypeInteger`, so match it.
            return format(arena, str_lit("i{}"), (int64_t)type->data.integer.width);
        case TYPE_KIND_FLOAT:
            if (type->data.floating.is_bfloat && type->data.floating.width == 16) {
                return str_lit("bf16");
            }
            return format(arena, str_lit("f{}"), (int64_t)type->data.floating.width);
        case TYPE_KIND_TENSOR: {
            MLIR_TypeHandle elem_h = type->data.shaped.element_type;
            if (elem_h != MLIR_INVALID_HANDLE) {
                string elem_str = MLIR_GetTypeString(ctx, elem_h);
                if (type->data.shaped.rank > 0 && type->data.shaped.shape) {
                    strbuf shape_str = strbuf_make();
                    for (uint32_t i = 0; i < type->data.shaped.rank; i++) {
                        int64_t dim = type->data.shaped.shape[i];
                        if (dim < 0) {
                            strbuf_append(arena, &shape_str, str_lit("?x"));
                        } else {
                            strbuf_append(arena, &shape_str, format(arena, str_lit("{}x"), dim));
                        }
                    }
                    return format(arena, str_lit("tensor<{}{}>"), strbuf_to_string(shape_str), elem_str);
                } else {
                    return format(arena, str_lit("tensor<{}>"), elem_str);
                }
            }
            return str_lit("tensor<?>");
        }
        case TYPE_KIND_MEMREF: {
            MLIR_TypeHandle elem_h = type->data.shaped.element_type;
            if (elem_h != MLIR_INVALID_HANDLE) {
                string elem_str = MLIR_GetTypeString(ctx, elem_h);
                if (type->data.shaped.rank > 0 && type->data.shaped.shape) {
                    strbuf shape_str = strbuf_make();
                    for (uint32_t i = 0; i < type->data.shaped.rank; i++) {
                        int64_t dim = type->data.shaped.shape[i];
                        if (dim < 0) {
                            strbuf_append(arena, &shape_str, str_lit("?x"));
                        } else {
                            strbuf_append(arena, &shape_str, format(arena, str_lit("{}x"), dim));
                        }
                    }
                    return format(arena, str_lit("memref<{}{}>"), strbuf_to_string(shape_str), elem_str);
                } else {
                    return format(arena, str_lit("memref<{}>"), elem_str);
                }
            }
            return str_lit("memref<?>");
        }
        case TYPE_KIND_POINTER: {
            MLIR_TypeHandle elem_h = type->data.pointer.element_type;
            if (elem_h != MLIR_INVALID_HANDLE) {
                string elem_str = MLIR_GetTypeString(ctx, elem_h);
                if (type->data.pointer.has_address_space) {
                    return format(arena, str_lit("!tt.ptr<{}, {}>"), elem_str, (int64_t)type->data.pointer.address_space);
                } else {
                    return format(arena, str_lit("!tt.ptr<{}>"), elem_str);
                }
            }
            return str_lit("!tt.ptr<?>");
        }
        case TYPE_KIND_INDEX:
            return str_lit("index");
        case TYPE_KIND_FUNCTION: {
            strbuf in_str = strbuf_make();
            for (size_t i = 0; i < type->data.function.n_inputs; i++) {
                if (i > 0) strbuf_append(arena, &in_str, str_lit(", "));
                strbuf_append(arena, &in_str,
                              MLIR_GetTypeString(ctx, type->data.function.inputs[i]));
            }
            string out_str = str_lit("");
            size_t nr = type->data.function.n_results;
            if (nr == 1) {
                out_str = MLIR_GetTypeString(ctx, type->data.function.results[0]);
            } else {
                strbuf body = strbuf_make();
                for (size_t i = 0; i < nr; i++) {
                    if (i > 0) strbuf_append(arena, &body, str_lit(", "));
                    strbuf_append(arena, &body,
                                  MLIR_GetTypeString(ctx, type->data.function.results[i]));
                }
                out_str = format(arena, str_lit("({})"), strbuf_to_string(body));
            }
            return format(arena, str_lit("({}) -> {}"), strbuf_to_string(in_str), out_str);
        }
        case TYPE_KIND_LLVM_PTR:
            if (type->data.llvm_pointer.address_space == 0) return str_lit("!llvm.ptr");
            return format(arena, str_lit("!llvm.ptr<{}>"),
                          (int64_t)type->data.llvm_pointer.address_space);
        case TYPE_KIND_LLVM_VOID:
            return str_lit("!llvm.void");
        case TYPE_KIND_LLVM_ARRAY: {
            string elem = MLIR_GetTypeString(ctx, type->data.llvm_array.element);
            return format(arena, str_lit("!llvm.array<{} x {}>"),
                          (int64_t)type->data.llvm_array.count, elem);
        }
        case TYPE_KIND_LLVM_STRUCT: {
            // Anonymous: !llvm.struct<(T1, T2)>; identified: !llvm.struct<"name", (T1, T2)>.
            // Within nested struct fields MLIR omits the outer "!llvm." prefix —
            // but the translator's print_llvm_type_text accepts both, so we
            // always emit the canonical form here for simplicity.
            strbuf body = strbuf_make();
            strbuf_append(arena, &body, str_lit("("));
            for (size_t i = 0; i < type->data.llvm_struct.n_fields; i++) {
                if (i > 0) strbuf_append(arena, &body, str_lit(", "));
                strbuf_append(arena, &body,
                              MLIR_GetTypeString(ctx, type->data.llvm_struct.fields[i]));
            }
            strbuf_append(arena, &body, str_lit(")"));
            if (type->data.llvm_struct.name.size > 0) {
                return format(arena, str_lit("!llvm.struct<\"{}\", {}>"),
                              type->data.llvm_struct.name, strbuf_to_string(body));
            }
            return format(arena, str_lit("!llvm.struct<{}>"), strbuf_to_string(body));
        }
        case TYPE_KIND_LLVM_FUNCTION: {
            string ret = MLIR_GetTypeString(ctx, type->data.llvm_function.result);
            strbuf body = strbuf_make();
            for (size_t i = 0; i < type->data.llvm_function.n_inputs; i++) {
                if (i > 0) strbuf_append(arena, &body, str_lit(", "));
                strbuf_append(arena, &body,
                              MLIR_GetTypeString(ctx, type->data.llvm_function.inputs[i]));
            }
            if (type->data.llvm_function.is_var_arg) {
                if (type->data.llvm_function.n_inputs) {
                    strbuf_append(arena, &body, str_lit(", "));
                }
                strbuf_append(arena, &body, str_lit("..."));
            }
            return format(arena, str_lit("!llvm.func<{} ({})>"), ret, strbuf_to_string(body));
        }
        default:
            return str_lit("unknown");
    }
}

// Type creation
MLIR_TypeHandle MLIR_CreateTypeInteger(MLIR_Context *ctx, uint32_t width, bool is_signed) {
    // MLIR integer types are signless by default. The public API takes an
    // `is_signed` flag for forward-compat, but the upstream backend ignores
    // it (always constructs a signless `IntegerType`). Match that so both
    // backends produce the same canonical type identity for `i{w}` and the
    // wasm pipeline / lowering passes can compare types via handle equality.
    (void)is_signed;
    IR_Type t = {0};
    t.kind = TYPE_KIND_INTEGER;
    t.data.integer.width = width;
    t.data.integer.is_signed = false;
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeFloat(MLIR_Context *ctx, uint32_t width, bool is_bfloat) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_FLOAT;
    t.data.floating.width = width;
    t.data.floating.is_bfloat = is_bfloat;
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeIndex(MLIR_Context *ctx) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_INDEX;
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeUnknown(MLIR_Context *ctx) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_UNKNOWN;
    return intern_type(ctx, t);
}

// Point a candidate shaped type at the caller's shape; intern_type copies it.
static void set_shape(IR_Type *type, const int64_t *shape, size_t rank) {
    type->data.shaped.rank = (uint32_t)rank;
    type->data.shaped.shape = (rank > 0 && shape) ? (int64_t *)shape : NULL;
}

MLIR_TypeHandle MLIR_CreateTypeTensor(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_TENSOR;
    t.data.shaped.element_type = element_type;
    set_shape(&t, shape, rank);
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeMemref(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_MEMREF;
    t.data.shaped.element_type = element_type;
    set_shape(&t, shape, rank);
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypePointer(MLIR_Context *ctx, MLIR_TypeHandle element_type, bool has_address_space, uint32_t address_space) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_POINTER;
    t.data.pointer.element_type = element_type;
    t.data.pointer.has_address_space = has_address_space;
    t.data.pointer.address_space = address_space;
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeLLVMPointer(MLIR_Context *ctx) {
    return MLIR_CreateTypeLLVMPointerInAddressSpace(ctx, 0);
}

MLIR_TypeHandle MLIR_CreateTypeLLVMPointerInAddressSpace(
    MLIR_Context *ctx, uint32_t address_space) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_LLVM_PTR;
    t.data.llvm_pointer.address_space = address_space;
    return intern_type(ctx, t);
}

// Identified LLVM structs are interned by name within a context: the same
// printed reference (`!llvm.struct<"foo", ...>`) must resolve to the same
// type handle across all uses. A linear scan of the context's table by name
// (cheap given the small N in tinyc programs). The name is copied into the
// persistent arena with the type.
static MLIR_TypeHandle intern_llvm_struct(MLIR_Context *ctx, string name) {
    for (size_t i = 0; i < ctx->n_structs; i++) {
        if (ctx->struct_names[i].size == name.size &&
            memcmp(ctx->struct_names[i].str, name.str, name.size) == 0) {
            return ctx->struct_handles[i];
        }
    }
    Arena *a = ctx->persistent;
    if (ctx->n_structs == ctx->cap_structs) {
        size_t nc = ctx->cap_structs ? ctx->cap_structs * 2 : 32;
        MLIR_TypeHandle *nh = arena_new_array(a, MLIR_TypeHandle, nc);
        string *nn = arena_new_array(a, string, nc);
        if (ctx->n_structs) {
            memcpy(nh, ctx->struct_handles, ctx->n_structs * sizeof(MLIR_TypeHandle));
            memcpy(nn, ctx->struct_names, ctx->n_structs * sizeof(string));
        }
        ctx->struct_handles = nh;
        ctx->struct_names = nn;
        ctx->cap_structs = nc;
    }
    string owned = name;
    if (name.size > 0) {
        char *p = arena_new_array(a, char, name.size);
        memcpy(p, name.str, name.size);
        owned.str = p;
    }
    IR_Type *t = arena_new(a, IR_Type);
    *t = (IR_Type){0};
    t->kind = TYPE_KIND_LLVM_STRUCT;
    t->data.llvm_struct.name = owned;
    MLIR_TypeHandle h = (MLIR_TypeHandle)t;
    ctx->struct_handles[ctx->n_structs] = h;
    ctx->struct_names[ctx->n_structs] = owned;
    ctx->n_structs++;
    return h;
}

MLIR_TypeHandle MLIR_CreateTypeLLVMStructIdentified(MLIR_Context *ctx, string name) {
    return intern_llvm_struct(ctx, name);
}

void MLIR_SetTypeLLVMStructBody(MLIR_Context *ctx, MLIR_TypeHandle struct_ty,
                                 const MLIR_TypeHandle *fields, size_t n_fields) {
    IR_Type *t = resolve_type(struct_ty);
    if (!t || t->kind != TYPE_KIND_LLVM_STRUCT) return;
    if (t->data.llvm_struct.body_set) return; // upstream silently ignores re-set
    if (n_fields > 0) {
        MLIR_TypeHandle *buf = arena_new_array(ctx->persistent, MLIR_TypeHandle, n_fields);
        memcpy(buf, fields, n_fields * sizeof(MLIR_TypeHandle));
        t->data.llvm_struct.fields = buf;
    }
    t->data.llvm_struct.n_fields = n_fields;
    t->data.llvm_struct.body_set = true;
}

MLIR_TypeHandle MLIR_CreateTypeLLVMArray(MLIR_Context *ctx, MLIR_TypeHandle elem, uint64_t count) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_LLVM_ARRAY;
    t.data.llvm_array.element = elem;
    t.data.llvm_array.count = count;
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeLLVMFunction(MLIR_Context *ctx,
                                             MLIR_TypeHandle result,
                                             const MLIR_TypeHandle *inputs,
                                             size_t n_inputs,
                                             bool is_var_arg) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_LLVM_FUNCTION;
    t.data.llvm_function.result = result;
    t.data.llvm_function.is_var_arg = is_var_arg;
    if (n_inputs > 0) {
        t.data.llvm_function.inputs = (MLIR_TypeHandle *)inputs;
        t.data.llvm_function.n_inputs = n_inputs;
    }
    return intern_type(ctx, t);
}

MLIR_TypeHandle MLIR_CreateTypeLLVMVoid(MLIR_Context *ctx) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_LLVM_VOID;
    return intern_type(ctx, t);
}

// LLVM-dialect global helpers. Construct `llvm.mlir.global` ops with the
// attributes the LLVM-IR translator looks for (sym_name, global_type,
// linkage, constant, value). Both helpers return the op so the caller
// can append it to the module body. MLIR_CreateLLVMGlobal optionally
// hands back the entry block of the initializer region for region-init
// globals.
static MLIR_AttributeHandle make_llvm_linkage_attr(MLIR_Context *ctx, string linkage_kind) {
    // Mirror the upstream printer: "#llvm.linkage<<kind>>".
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = str_lit("linkage");
    strbuf body = strbuf_make();
    strbuf_append(ctx->arena, &body, str_lit("#llvm.linkage<"));
    strbuf_append(ctx->arena, &body, linkage_kind);
    strbuf_append(ctx->arena, &body, str_lit(">"));
    a.data.string_value = strbuf_to_string(body);
    return alloc_attr_obj(ctx, a);
}

MLIR_OpHandle MLIR_CreateLLVMGlobalString(MLIR_Context *ctx, string sym_name,
                                          string bytes, MLIR_LocationHandle loc) {
    MLIR_TypeHandle i8 = MLIR_CreateTypeInteger(ctx, 8, false);
    MLIR_TypeHandle arr = MLIR_CreateTypeLLVMArray(ctx, i8, bytes.size);
    MLIR_AttributeHandle attrs[5];
    attrs[0] = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), sym_name);
    attrs[1] = MLIR_CreateAttributeType(ctx, str_lit("global_type"), arr);
    attrs[2] = make_llvm_linkage_attr(ctx, str_lit("private"));
    attrs[3] = MLIR_CreateAttributeBool(ctx, str_lit("constant"), true);
    attrs[4] = MLIR_CreateAttributeString(ctx, str_lit("value"), bytes);
    return MLIR_CreateOp(ctx, OP_TYPE_LLVM_MLIR_GLOBAL,
                         attrs, 5, NULL, 0, NULL, 0, NULL, 0,
                         NULL, 0, loc, MLIR_INVALID_HANDLE,
                         str_lit(""), -1);
}

// Aggregate-array global with a raw-bytes initializer:
//   llvm.mlir.global internal @<sym>("<bytes>") : !llvm.array<N x iX>
// Used for `static const T arr[N] = { v1, v2, ... };` at file scope.
// The `bytes` buffer holds the packed little-endian element bytes
// (size must equal N * sizeof(elem)).
// Backends interpret the STRING `value` attribute as raw initializer
// bytes (see lower_global in mlir_llvm_to_wasmssa.c and emit_global in
// mlir_translate_to_llvm_ir.c).
MLIR_OpHandle MLIR_CreateLLVMGlobalArrayInit(MLIR_Context *ctx, string sym_name,
                                              MLIR_TypeHandle array_ty,
                                              bool is_constant,
                                              string bytes,
                                              MLIR_LocationHandle loc) {
    MLIR_AttributeHandle attrs[5];
    size_t na = 0;
    attrs[na++] = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), sym_name);
    attrs[na++] = MLIR_CreateAttributeType(ctx, str_lit("global_type"), array_ty);
    attrs[na++] = make_llvm_linkage_attr(ctx, str_lit("internal"));
    if (is_constant) {
        attrs[na++] = MLIR_CreateAttributeBool(ctx, str_lit("constant"), true);
    }
    attrs[na++] = MLIR_CreateAttributeString(ctx, str_lit("value"), bytes);
    return MLIR_CreateOp(ctx, OP_TYPE_LLVM_MLIR_GLOBAL,
                         attrs, na, NULL, 0, NULL, 0, NULL, 0,
                         NULL, 0, loc, MLIR_INVALID_HANDLE,
                         str_lit(""), -1);
}

MLIR_OpHandle MLIR_CreateLLVMGlobal(MLIR_Context *ctx, string sym_name,
                                    MLIR_TypeHandle elem_ty, bool is_constant,
                                    int init_kind, int64_t init_int, double init_float,
                                    MLIR_BlockHandle *out_init_block,
                                    MLIR_LocationHandle loc) {
    // init_kind: 0=integer, 1=float, 2=region, 3=zero-init/no-init, 4=external
    MLIR_AttributeHandle attrs[6];
    size_t na = 0;
    attrs[na++] = MLIR_CreateAttributeString(ctx, str_lit("sym_name"), sym_name);
    attrs[na++] = MLIR_CreateAttributeType(ctx, str_lit("global_type"), elem_ty);
    string lk = (init_kind == 4) ? str_lit("external") : str_lit("internal");
    attrs[na++] = make_llvm_linkage_attr(ctx, lk);
    if (is_constant) attrs[na++] = MLIR_CreateAttributeBool(ctx, str_lit("constant"), true);
    if (init_kind == 0) {
        attrs[na++] = MLIR_CreateAttributeInteger(ctx, str_lit("value"), init_int, elem_ty);
    } else if (init_kind == 1) {
        attrs[na++] = MLIR_CreateAttributeFloat(ctx, str_lit("value"), init_float, elem_ty);
    }

    // Region for region-initialized globals (init_kind == 2): create one
    // empty block so the caller can populate it.
    MLIR_RegionHandle regs[1];
    size_t nr = 0;
    if (init_kind == 2) {
        MLIR_RegionHandle r = MLIR_CreateRegion(ctx);
        MLIR_BlockHandle entry = MLIR_CreateBlock(ctx);
        MLIR_AppendRegionBlock(ctx, r, entry);
        if (out_init_block) *out_init_block = entry;
        regs[0] = r; nr = 1;
    } else if (out_init_block) {
        *out_init_block = MLIR_INVALID_HANDLE;
    }
    return MLIR_CreateOp(ctx, OP_TYPE_LLVM_MLIR_GLOBAL,
                         attrs, na, NULL, 0, NULL, 0, NULL, 0,
                         nr ? regs : NULL, nr, loc, MLIR_INVALID_HANDLE,
                         str_lit(""), -1);
}

MLIR_TypeHandle MLIR_CreateTypeFunction(MLIR_Context *ctx,
                                         const MLIR_TypeHandle *inputs, size_t n_inputs,
                                         const MLIR_TypeHandle *results, size_t n_results) {
    IR_Type t = {0};
    t.kind = TYPE_KIND_FUNCTION;
    if (n_inputs > 0 && inputs) {
        t.data.function.inputs = (MLIR_TypeHandle *)inputs;
        t.data.function.n_inputs = n_inputs;
    }
    if (n_results > 0 && results) {
        t.data.function.results = (MLIR_TypeHandle *)results;
        t.data.function.n_results = n_results;
    }
    return intern_type(ctx, t);
}

size_t MLIR_GetTypeFunctionNumInputs(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t) return 0;
    if (t->kind == TYPE_KIND_FUNCTION) return t->data.function.n_inputs;
    if (t->kind == TYPE_KIND_LLVM_FUNCTION) return t->data.llvm_function.n_inputs;
    return 0;
}

MLIR_TypeHandle MLIR_GetTypeFunctionInput(MLIR_TypeHandle th, size_t idx) {
    API_CHECK(idx < MLIR_GetTypeFunctionNumInputs(th), "MLIR_GetTypeFunctionInput", "index out of range");
    IR_Type *t = resolve_type(th);
    if (!t) return MLIR_INVALID_HANDLE;
    if (t->kind == TYPE_KIND_FUNCTION) {
        if (idx >= t->data.function.n_inputs || !t->data.function.inputs) return MLIR_INVALID_HANDLE;
        return t->data.function.inputs[idx];
    }
    if (t->kind == TYPE_KIND_LLVM_FUNCTION) {
        if (idx >= t->data.llvm_function.n_inputs || !t->data.llvm_function.inputs) return MLIR_INVALID_HANDLE;
        return t->data.llvm_function.inputs[idx];
    }
    return MLIR_INVALID_HANDLE;
}

size_t MLIR_GetTypeFunctionNumResults(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t) return 0;
    if (t->kind == TYPE_KIND_FUNCTION) return t->data.function.n_results;
    if (t->kind == TYPE_KIND_LLVM_FUNCTION) {
        // LLVMFunctionType always has exactly one return type; void is
        // represented as a separate LLVMVoid type. Surface 0 results in
        // the void case so callers can use the same "0 results means void"
        // convention as upstream's MLIR_GetTypeFunctionNumResults.
        IR_Type *r = resolve_type(t->data.llvm_function.result);
        return (r && r->kind == TYPE_KIND_LLVM_VOID) ? 0 : 1;
    }
    return 0;
}

MLIR_TypeHandle MLIR_GetTypeFunctionResult(MLIR_TypeHandle th, size_t idx) {
    API_CHECK(idx < MLIR_GetTypeFunctionNumResults(th), "MLIR_GetTypeFunctionResult", "index out of range");
    IR_Type *t = resolve_type(th);
    if (!t) return MLIR_INVALID_HANDLE;
    if (t->kind == TYPE_KIND_FUNCTION) {
        if (idx >= t->data.function.n_results || !t->data.function.results) return MLIR_INVALID_HANDLE;
        return t->data.function.results[idx];
    }
    if (t->kind == TYPE_KIND_LLVM_FUNCTION) {
        if (idx != 0) return MLIR_INVALID_HANDLE;
        IR_Type *r = resolve_type(t->data.llvm_function.result);
        if (r && r->kind == TYPE_KIND_LLVM_VOID) return MLIR_INVALID_HANDLE;
        return t->data.llvm_function.result;
    }
    return MLIR_INVALID_HANDLE;
}

bool MLIR_GetTypeFunctionIsVarArg(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t) return false;
    if (t->kind == TYPE_KIND_LLVM_FUNCTION) return t->data.llvm_function.is_var_arg;
    return false;
}

MLIR_TypeHandle MLIR_GetTypeShapedElement(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t) return MLIR_INVALID_HANDLE;
    if (t->kind == TYPE_KIND_TENSOR || t->kind == TYPE_KIND_MEMREF) {
        return t->data.shaped.element_type;
    }
    return MLIR_INVALID_HANDLE;
}

bool MLIR_IsTypeLLVMStruct(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_LLVM_STRUCT;
}
size_t MLIR_GetTypeLLVMStructNumFields(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t || t->kind != TYPE_KIND_LLVM_STRUCT) return 0;
    return t->data.llvm_struct.n_fields;
}
MLIR_TypeHandle MLIR_GetTypeLLVMStructField(MLIR_TypeHandle th, size_t idx) {
    API_CHECK(idx < MLIR_GetTypeLLVMStructNumFields(th), "MLIR_GetTypeLLVMStructField", "index out of range");
    IR_Type *t = resolve_type(th);
    if (!t || t->kind != TYPE_KIND_LLVM_STRUCT) return MLIR_INVALID_HANDLE;
    if (idx >= t->data.llvm_struct.n_fields) return MLIR_INVALID_HANDLE;
    return t->data.llvm_struct.fields[idx];
}
bool MLIR_IsTypeLLVMArray(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_LLVM_ARRAY;
}
MLIR_TypeHandle MLIR_GetTypeLLVMArrayElement(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t || t->kind != TYPE_KIND_LLVM_ARRAY) return MLIR_INVALID_HANDLE;
    return t->data.llvm_array.element;
}
uint64_t MLIR_GetTypeLLVMArrayNumElements(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    if (!t || t->kind != TYPE_KIND_LLVM_ARRAY) return 0;
    return t->data.llvm_array.count;
}

uint32_t MLIR_GetTypeLLVMPointerAddressSpace(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return (t && t->kind == TYPE_KIND_LLVM_PTR)
        ? t->data.llvm_pointer.address_space : 0;
}

// Attribute creation
MLIR_AttributeHandle MLIR_CreateAttributeInteger(MLIR_Context *ctx, string name, int64_t value, MLIR_TypeHandle type) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_INTEGER;
    a.name = arena_dup_str(ctx, name);
    a.data.integer_value = value;
    a.type = type;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeString(MLIR_Context *ctx, string name, string value) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = arena_dup_str(ctx, name);
    a.data.string_value = arena_dup_str(ctx, value);
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeFloat(MLIR_Context *ctx, string name, double value, MLIR_TypeHandle type) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_FLOAT;
    a.name = arena_dup_str(ctx, name);
    a.data.float_value = value;
    a.type = type;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeBool(MLIR_Context *ctx, string name, bool value) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_BOOL;
    a.name = arena_dup_str(ctx, name);
    a.data.bool_value = value;
    return alloc_attr_obj(ctx, a);
}

// Copy an array/dict attribute's element handles into the context's arena.
// Callers routinely build `elements` on the stack (e.g. lower_func_func's
// `passthrough`), so storing their pointer would leave the attribute
// dangling once that frame returns. The upstream backend copies the
// elements into its ArrayAttr / DictionaryAttr, so this matches it.
static MLIR_AttributeHandle *dup_attr_elements(MLIR_Context *ctx,
                                               const MLIR_AttributeHandle *elements,
                                               size_t count) {
    if (count == 0 || !elements || !ctx || !ctx->arena) return NULL;
    MLIR_AttributeHandle *buf = arena_new_array(ctx->arena, MLIR_AttributeHandle, count);
    memcpy(buf, elements, count * sizeof(MLIR_AttributeHandle));
    return buf;
}

MLIR_AttributeHandle MLIR_CreateAttributeArray(MLIR_Context *ctx, string name, MLIR_AttributeHandle *elements, size_t count) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_ARRAY;
    a.name = arena_dup_str(ctx, name);
    a.data.array.elements = dup_attr_elements(ctx, elements, count);
    a.data.array.count = a.data.array.elements ? count : 0;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeDict(MLIR_Context *ctx, string name, MLIR_AttributeHandle *elements, size_t count) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_DICT;
    a.name = arena_dup_str(ctx, name);
    a.data.array.elements = dup_attr_elements(ctx, elements, count);
    a.data.array.count = a.data.array.elements ? count : 0;
    return alloc_attr_obj(ctx, a);
}

MLIR_AttributeHandle MLIR_CreateAttributeType(MLIR_Context *ctx, string name, MLIR_TypeHandle type) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_TYPE;
    a.name = arena_dup_str(ctx, name);
    a.data.type_value = type;
    return alloc_attr_obj(ctx, a);
}

// Native backend stores SymbolRef as a plain string attribute. The
// upstream printer renders symbol references as `@name`, so we store
// the same form here — keeps MLIR_GetAttributeAsString consistent
// across backends (the LLVM-IR translator depends on this for callee
// emission).
MLIR_AttributeHandle MLIR_CreateAttributeSymbolRef(MLIR_Context *ctx, string name, string value) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = arena_dup_str(ctx, name);
    string with_at = format(ctx->arena, str_lit("@{}"), value);
    a.data.string_value = with_at;
    return alloc_attr_obj(ctx, a);
}

// DenseI32 array attribute. The native LLVM-IR translator reads this
// via MLIR_GetAttributeAsString and parses the printed form, so we
// stash a string matching upstream's printer ("array<i32: v0, v1, ...>").
MLIR_AttributeHandle MLIR_CreateAttributeDenseI32Array(MLIR_Context *ctx, string name,
                                                       const int32_t *values, size_t count) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = arena_dup_str(ctx, name);
    Arena *arena = ctx->arena;
    strbuf s = strbuf_make();
    strbuf_append(arena, &s, str_lit("array<i32"));
    for (size_t i = 0; i < count; i++) {
        strbuf_append(arena, &s, i == 0 ? str_lit(": ") : str_lit(", "));
        strbuf_append(arena, &s, format(arena, str_lit("{}"), (int64_t)values[i]));
    }
    strbuf_append(arena, &s, str_lit(">"));
    a.data.string_value = strbuf_to_string(s);
    return alloc_attr_obj(ctx, a);
}

// DenseI64 array attribute. Same string-encoded shape as DenseI32 but
// with `i64` element type. Used by `scf.index_switch`'s `cases` attr.
MLIR_AttributeHandle MLIR_CreateAttributeDenseI64Array(MLIR_Context *ctx, string name,
                                                       const int64_t *values, size_t count) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = arena_dup_str(ctx, name);
    Arena *arena = ctx->arena;
    strbuf s = strbuf_make();
    strbuf_append(arena, &s, str_lit("array<i64"));
    for (size_t i = 0; i < count; i++) {
        strbuf_append(arena, &s, i == 0 ? str_lit(": ") : str_lit(", "));
        strbuf_append(arena, &s, format(arena, str_lit("{}"), values[i]));
    }
    strbuf_append(arena, &s, str_lit(">"));
    a.data.string_value = strbuf_to_string(s);
    return alloc_attr_obj(ctx, a);
}

// LLVM linkage attribute. Native backend stores the printable form
// (e.g. "#llvm.linkage<internal>") in a string attr — this matches the
// upstream backend's MLIR_GetAttributeAsString result, so the native
// LLVM-IR translator can recognize it the same way for both backends.
MLIR_AttributeHandle MLIR_CreateAttributeLLVMLinkageInternal(MLIR_Context *ctx, string name) {
    IR_Attribute a = {0};
    a.kind = ATTR_KIND_STRING;
    a.name = arena_dup_str(ctx, name);
    a.data.string_value = str_lit("#llvm.linkage<internal>");
    return alloc_attr_obj(ctx, a);
}

// Value creation
MLIR_ValueHandle MLIR_CreateValueBlockArg(MLIR_Context *ctx, string register_name, uint32_t result_index, MLIR_TypeHandle type, MLIR_LocationHandle location) {
    API_CHECK(ctx != NULL, "MLIR_CreateValueBlockArg", "context is NULL");
    uint32_t v = value_create(ctx, 0, result_index, type, location, register_name);
    VALS(ctx)[v].index |= VAL_BLOCK_ARG;
    return VAL_H(v);
}

MLIR_ValueHandle MLIR_CreateValueOpResult(MLIR_Context *ctx, MLIR_OpHandle def, uint32_t result_index, MLIR_TypeHandle type, string register_name, MLIR_LocationHandle location) {
    API_CHECK(ctx != NULL, "MLIR_CreateValueOpResult", "context is NULL");
    uint32_t d = OP_IDX(ctx, def, "MLIR_CreateValueOpResult");
    CHECK_NOT_ERASED(ctx, d, "MLIR_CreateValueOpResult");
    return VAL_H(value_create(ctx, d, result_index, type, location, register_name));
}

// Block and Region creation
MLIR_BlockHandle MLIR_CreateBlock(MLIR_Context *ctx) {
    API_CHECK(ctx != NULL, "MLIR_CreateBlock", "context is NULL");
    uint32_t b = vec_push(&ctx->blocks, 1);
    memset(&BLOCKS(ctx)[b], 0, sizeof(IR_Block));
    return BLOCK_H(b);
}

MLIR_RegionHandle MLIR_CreateRegion(MLIR_Context *ctx) {
    API_CHECK(ctx != NULL, "MLIR_CreateRegion", "context is NULL");
    uint32_t r = vec_push(&ctx->regions, 1);
    memset(&REGIONS(ctx)[r], 0, sizeof(IR_Region));
    return REGION_H(r);
}

// Unless they are already last, the op's attributes move to the end of the
// ptrs vector, one longer; the old entries are left unused.
void MLIR_AppendOpAttribute(MLIR_Context *ctx, MLIR_OpHandle oh, MLIR_AttributeHandle attr) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_AppendOpAttribute");
    CHECK_OP(ctx, o, "MLIR_AppendOpAttribute");
    API_CHECK(attr != MLIR_INVALID_HANDLE, "MLIR_AppendOpAttribute", "attribute handle is NULL");
    uint32_t n = OPS(ctx)[o].n_attributes;
    API_CHECK(n < 0xffff, "MLIR_AppendOpAttribute", "too many attributes");
    uint32_t old = OPS(ctx)[o].first_attr;
    if (old + n == ctx->ptrs.n) {
        // Already last (the common case: attributes appended right after
        // creation): grow in place.
        vec_push(&ctx->ptrs, 1);
        PTRS(ctx)[old + n] = (uintptr_t)attr;
        OPS(ctx)[o].n_attributes = (uint16_t)(n + 1);
        return;
    }
    uint32_t first = vec_push(&ctx->ptrs, n + 1);
    uintptr_t *p = PTRS(ctx);
    for (uint32_t i = 0; i < n; i++) p[first + i] = p[old + i];
    p[first + n] = (uintptr_t)attr;
    OPS(ctx)[o].first_attr = first;
    OPS(ctx)[o].n_attributes = (uint16_t)(n + 1);
}

// Attribute accessors
MLIR_AttrKind MLIR_GetAttributeKind(MLIR_AttributeHandle ah) {
    API_CHECK(ah != MLIR_INVALID_HANDLE, "MLIR_GetAttributeKind", "attribute handle is NULL");
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr) return MLIR_ATTR_KIND_INTEGER;
    switch (attr->kind) {
        case ATTR_KIND_INTEGER: return MLIR_ATTR_KIND_INTEGER;
        case ATTR_KIND_FLOAT:   return MLIR_ATTR_KIND_FLOAT;
        case ATTR_KIND_STRING:  return MLIR_ATTR_KIND_STRING;
        case ATTR_KIND_BOOL:    return MLIR_ATTR_KIND_BOOL;
        case ATTR_KIND_ARRAY:   return MLIR_ATTR_KIND_ARRAY;
        case ATTR_KIND_DICT:    return MLIR_ATTR_KIND_DICT;
        case ATTR_KIND_TYPE:    return MLIR_ATTR_KIND_TYPE;
        default:                return MLIR_ATTR_KIND_DICT;
    }
}

string MLIR_GetAttributeName(MLIR_AttributeHandle ah) {
    API_CHECK(ah != MLIR_INVALID_HANDLE, "MLIR_GetAttributeName", "attribute handle is NULL");
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->name : str_lit("");
}

// Native parser never produces "other" attrs; this is just for API parity.
string MLIR_GetAttributeAsString(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    (void)ctx;
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr) return str_lit("");
    // The native lowering/translation passes only consult this for a few
    // attributes whose printed form is well-defined (LLVM linkage, plain
    // strings). For everything else we return empty — callers must use
    // the typed accessors.
    if (attr->kind == ATTR_KIND_STRING) return attr->data.string_value;
    return str_lit("");
}

int64_t MLIR_GetAttributeInteger(MLIR_AttributeHandle ah) {
    CHECK_ATTR_KIND(ah, ((IR_Attribute *)ah)->kind == ATTR_KIND_INTEGER || ((IR_Attribute *)ah)->kind == ATTR_KIND_BOOL, "MLIR_GetAttributeInteger", "not an integer or bool attribute");
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.integer_value : 0;
}

string MLIR_GetAttributeString(MLIR_AttributeHandle ah) {
    CHECK_ATTR_KIND(ah, ((IR_Attribute *)ah)->kind == ATTR_KIND_STRING, "MLIR_GetAttributeString", "not a string attribute");
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.string_value : str_lit("");
}

double MLIR_GetAttributeFloat(MLIR_AttributeHandle ah) {
    CHECK_ATTR_KIND(ah, ((IR_Attribute *)ah)->kind == ATTR_KIND_FLOAT, "MLIR_GetAttributeFloat", "not a float attribute");
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.float_value : 0.0;
}

MLIR_TypeHandle MLIR_GetAttributeType(MLIR_AttributeHandle ah) {
    API_CHECK(ah != MLIR_INVALID_HANDLE, "MLIR_GetAttributeType", "attribute handle is NULL");
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr) return MLIR_INVALID_HANDLE;
    if (attr->kind != ATTR_KIND_INTEGER && attr->kind != ATTR_KIND_FLOAT)
        return MLIR_INVALID_HANDLE;
    return attr->type;
}

bool MLIR_GetAttributeBool(MLIR_AttributeHandle ah) {
    CHECK_ATTR_KIND(ah, ((IR_Attribute *)ah)->kind == ATTR_KIND_BOOL, "MLIR_GetAttributeBool", "not a bool attribute");
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.bool_value : false;
}

size_t MLIR_GetAttributeArraySize(MLIR_AttributeHandle ah) {
    CHECK_ATTR_KIND(ah, ((IR_Attribute *)ah)->kind == ATTR_KIND_ARRAY, "MLIR_GetAttributeArraySize", "not an array attribute");
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.array.count : 0;
}

MLIR_AttributeHandle MLIR_GetAttributeArrayElement(MLIR_AttributeHandle ah, size_t idx) {
    CHECK_ATTR_KIND(ah, ((IR_Attribute *)ah)->kind == ATTR_KIND_ARRAY, "MLIR_GetAttributeArrayElement", "not an array attribute");
    CHECK_INDEX(idx, ((IR_Attribute *)ah)->data.array.count, "MLIR_GetAttributeArrayElement");
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr || idx >= attr->data.array.count) return MLIR_INVALID_HANDLE;
    return attr->data.array.elements[idx];
}

size_t MLIR_GetAttributeDictSize(MLIR_AttributeHandle ah) {
    CHECK_ATTR_KIND(ah, ((IR_Attribute *)ah)->kind == ATTR_KIND_DICT, "MLIR_GetAttributeDictSize", "not a dictionary attribute");
    IR_Attribute *attr = resolve_attr(ah);
    return attr ? attr->data.array.count : 0;
}

MLIR_AttributeHandle MLIR_GetAttributeDictElement(MLIR_AttributeHandle ah, size_t idx) {
    CHECK_ATTR_KIND(ah, ((IR_Attribute *)ah)->kind == ATTR_KIND_DICT, "MLIR_GetAttributeDictElement", "not a dictionary attribute");
    CHECK_INDEX(idx, ((IR_Attribute *)ah)->data.array.count, "MLIR_GetAttributeDictElement");
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr || idx >= attr->data.array.count) return MLIR_INVALID_HANDLE;
    return attr->data.array.elements[idx];
}

MLIR_TypeHandle MLIR_GetAttributeTypeValue(MLIR_AttributeHandle ah) {
    API_CHECK(ah != MLIR_INVALID_HANDLE, "MLIR_GetAttributeTypeValue", "attribute handle is NULL");
    IR_Attribute *attr = resolve_attr(ah);
    if (!attr || attr->kind != ATTR_KIND_TYPE) return MLIR_INVALID_HANDLE;
    return attr->data.type_value;
}

// Value accessors
MLIR_ValueKind MLIR_GetValueKind(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_GetValueKind");
    API_CHECK(v != 0, "MLIR_GetValueKind", "value handle is NULL");
    return (VALS(ctx)[v].index & VAL_BLOCK_ARG) ? MLIR_VALUE_BLOCK_ARG : MLIR_VALUE_OP_RESULT;
}

MLIR_TypeHandle MLIR_GetValueType(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_GetValueType");
    API_CHECK(v != 0, "MLIR_GetValueType", "value handle is NULL");
    return VALS(ctx)[v].type;
}

string MLIR_GetValueRegisterName(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_GetValueRegisterName");
    uint32_t n = VALS(ctx)[v].name;
    return n ? NAMES(ctx)[n] : str_lit("");
}

uint32_t MLIR_GetValueResultIndex(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_GetValueResultIndex");
    API_CHECK(v != 0, "MLIR_GetValueResultIndex", "value handle is NULL");
    return VALS(ctx)[v].index & VAL_INDEX_MASK;
}

MLIR_OpHandle MLIR_GetValueDefiningOp(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_GetValueDefiningOp");
    API_CHECK(v != 0, "MLIR_GetValueDefiningOp", "value handle is NULL");
    IR_Value *p = &VALS(ctx)[v];
    return (p->index & VAL_BLOCK_ARG) ? MLIR_INVALID_HANDLE : OP_H(p->def);
}

MLIR_OpType MLIR_RegisterOpType(MLIR_Context *ctx, string name) {
    API_CHECK(ctx != NULL, "MLIR_RegisterOpType", "context is NULL");
    API_CHECK(name.size > 0, "MLIR_RegisterOpType", "empty op name");
    return op_type_table_register(&ctx->op_types, name);
}

MLIR_OpType MLIR_LookupOpType(MLIR_Context *ctx, string name) {
    API_CHECK(ctx != NULL, "MLIR_LookupOpType", "context is NULL");
    return op_type_table_lookup(&ctx->op_types, name);
}

string MLIR_GetOpTypeName(MLIR_Context *ctx, MLIR_OpType type) {
    API_CHECK(ctx != NULL, "MLIR_GetOpTypeName", "context is NULL");
    return op_type_table_name(&ctx->op_types, type);
}

// Location accessors
MLIR_LocationKind MLIR_GetLocationKind(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    if (!loc) return MLIR_LOC_UNKNOWN;
    switch (loc->kind) {
        case LOC_KIND_FILE: return MLIR_LOC_FILE;
        case LOC_KIND_NAME: return MLIR_LOC_NAME;
        case LOC_KIND_CALLSITE: return MLIR_LOC_CALLSITE;
        case LOC_KIND_FUSED: return MLIR_LOC_FUSED;
        case LOC_KIND_REF: return MLIR_LOC_REF;
        case LOC_KIND_UNKNOWN: default: return MLIR_LOC_UNKNOWN;
    }
}

string MLIR_GetLocationOriginalText(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->original_text : str_lit("");
}

string MLIR_GetLocationFileFilename(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->data.file.filename : str_lit("");
}

int MLIR_GetLocationFileLine(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->data.file.line : 0;
}

int MLIR_GetLocationFileColumn(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->data.file.column : 0;
}

string MLIR_GetLocationName(MLIR_LocationHandle lh) {
    IR_Location *loc = resolve_loc(lh);
    return loc ? loc->data.name.name : str_lit("");
}

MLIR_LocationHandle MLIR_CreateLocationUnknown(MLIR_Context *ctx, string original_text) {
    IR_Location l = {0};
    l.kind = LOC_KIND_UNKNOWN;
    l.original_text = original_text;
    return alloc_loc(ctx, l);
}

MLIR_LocationHandle MLIR_CreateLocationFile(MLIR_Context *ctx, string filename, int line, int column) {
    IR_Location l = {0};
    l.kind = LOC_KIND_FILE;
    l.data.file.filename = filename;
    l.data.file.line = line;
    l.data.file.column = column;
    if (ctx && ctx->arena) {
        l.original_text = format(ctx->arena, str_lit("loc({}:{}:{})"), filename, (int64_t)line, (int64_t)column);
    }
    return alloc_loc(ctx, l);
}

MLIR_LocationHandle MLIR_CreateLocationName(MLIR_Context *ctx, string name) {
    IR_Location l = {0};
    l.kind = LOC_KIND_NAME;
    l.data.name.name = name;
    if (ctx && ctx->arena) {
        l.original_text = format(ctx->arena, str_lit("loc(\"{}\")"), name);
    }
    return alloc_loc(ctx, l);
}

MLIR_LocationHandle MLIR_CreateLocationRef(MLIR_Context *ctx, int ref_id) {
    IR_Location l = {0};
    l.kind = LOC_KIND_REF;
    l.data.ref.ref_id = ref_id;
    if (ctx && ctx->arena) {
        l.original_text = format(ctx->arena, str_lit("loc(#loc{})"), (int64_t)ref_id);
    }
    return alloc_loc(ctx, l);
}

MLIR_TypeHandle MLIR_CreateTypeOpaque(MLIR_Context *ctx, string name) {
    (void)name;
    IR_Type t = {0};
    t.kind = TYPE_KIND_OPAQUE;
    return intern_type(ctx, t);
}

// Type introspection
bool MLIR_IsTypeInteger(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_INTEGER;
}

bool MLIR_IsTypeFloat(MLIR_TypeHandle th) {
    IR_Type *t = resolve_type(th);
    return t && t->kind == TYPE_KIND_FLOAT;
}

// IR mutation primitives.

void MLIR_ReplaceAllUsesOfValue(MLIR_Context *ctx,
                                MLIR_ValueHandle old_value,
                                MLIR_ValueHandle new_value) {
    uint32_t from = VAL_IDX(ctx, old_value, "MLIR_ReplaceAllUsesOfValue");
    uint32_t to = VAL_IDX(ctx, new_value, "MLIR_ReplaceAllUsesOfValue");
    if (from == 0 || from == to) return;
    IR_Operand *rec = OPNDS(ctx);
    IR_Value *vals = VALS(ctx);
    uint32_t head = vals[from].first_use;
    if (!head) return;
    // Repoint every use; keep the chain only if the new value has a use list.
    bool keep = to != 0 && (vals[to].index & VAL_TRACKED);
    uint32_t tail = head;
    for (uint32_t u = head; u; ) {
        uint32_t next = rec[u].next;
        rec[u].value = to;
        if (!keep) rec[u].next = rec[u].prev = 0;
        tail = u;
        u = next;
    }
    if (keep) {
        // Splice the chain in front of the new value's uses.
        rec[tail].next = vals[to].first_use;
        if (vals[to].first_use) rec[vals[to].first_use].prev = tail;
        vals[to].first_use = head;
        vals[to].n_uses += vals[from].n_uses;
    }
    vals[from].first_use = 0;
    vals[from].n_uses = 0;
}

// Removes op `o` from its parent block's op list (O(n) compaction).
static void detach_from_block(MLIR_Context *ctx, uint32_t o) {
    uint32_t b = OPS(ctx)[o].parent_block;
    if (b) {
        IR_Block *blk = &BLOCKS(ctx)[b];
        u32_remove(blk->ops, &blk->n_ops, o);
    }
    OPS(ctx)[o].parent_block = 0;
}

// Unlinks every operand use of op `o` and of the ops nested in its regions,
// as upstream's Operation::erase does when it destroys them. Under
// MLIR_API_CHECKS the ops are also marked erased.
static void drop_op_uses(MLIR_Context *ctx, uint32_t o) {
    IR_Op *op = &OPS(ctx)[o];
    uint32_t first = op->first_operand;
    uint32_t n_all = op->n_operands + op->n_succ_operands;
    if (op->flags & OPF_TRACKED) {
        for (uint32_t i = 0; i < n_all; i++) {
            uint32_t v = OPNDS(ctx)[first + i].value;
            if (operand_linked(ctx, o, v)) use_unlink(ctx, first + i);
            OPNDS(ctx)[first + i].value = 0;
        }
    }
    for (uint32_t r = 0; r < op->n_regions; r++) {
        IR_Region *reg = &REGIONS(ctx)[EXTRA(ctx)[op->first_extra + r]];
        for (uint32_t bi = 0; bi < reg->n_blocks; bi++) {
            IR_Block *blk = &BLOCKS(ctx)[reg->blocks[bi]];
            for (uint32_t k = 0; k < blk->n_ops; k++) drop_op_uses(ctx, blk->ops[k]);
        }
    }
#ifdef MLIR_API_CHECKS
    OPS(ctx)[o].op_type = OP_TYPE_ERASED_SENTINEL;
#endif
}

#ifdef MLIR_API_CHECKS
// Erasing requires that nothing outside the erased subtree uses a value
// defined inside it (mlir_api.h). These walks check it for the values that
// track uses; without def-use tracking there is nothing to check.
// O(size of the subtree), checked builds only.

// True if op `owner` is `root` or nested in it.
static bool op_is_inside(MLIR_Context *ctx, uint32_t owner, uint32_t root) {
    uint32_t cur = owner;
    while (cur != 0) {
        if (cur == root) return true;
        uint32_t b = OPS(ctx)[cur].parent_block;
        if (b == 0) return false;
        uint32_t r = BLOCKS(ctx)[b].parent_region;
        cur = r ? REGIONS(ctx)[r].parent_op : 0;
    }
    return false;
}

static bool value_used_outside(MLIR_Context *ctx, uint32_t v, uint32_t root) {
    for (uint32_t u = VALS(ctx)[v].first_use; u; u = OPNDS(ctx)[u].next)
        if (!op_is_inside(ctx, OPNDS(ctx)[u].owner, root)) return true;
    return false;
}

// True if a value defined in op `o` or nested in it is used outside `root`.
static bool op_used_outside(MLIR_Context *ctx, uint32_t o, uint32_t root) {
    IR_Op *op = &OPS(ctx)[o];
    for (uint32_t i = 0; i < op->n_results; i++)
        if (value_used_outside(ctx, op->first_result + i, root)) return true;
    for (uint32_t r = 0; r < op->n_regions; r++) {
        IR_Region *reg = &REGIONS(ctx)[EXTRA(ctx)[op->first_extra + r]];
        for (uint32_t bi = 0; bi < reg->n_blocks; bi++) {
            IR_Block *b = &BLOCKS(ctx)[reg->blocks[bi]];
            for (uint32_t i = 0; i < b->n_args; i++)
                if (value_used_outside(ctx, b->args[i], root)) return true;
            for (uint32_t k = 0; k < b->n_ops; k++)
                if (op_used_outside(ctx, b->ops[k], root)) return true;
        }
    }
    return false;
}
#endif

// The op's slot stays (erased ops are never reused), but the handle is
// dead: see mlir_api.h.
void MLIR_EraseOp(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_EraseOp");
    CHECK_OP(ctx, o, "MLIR_EraseOp");
    if (!o) return;
#ifdef MLIR_API_CHECKS
    API_CHECK(!op_used_outside(ctx, o, o), "MLIR_EraseOp",
              "a value defined in the op or nested in it is used outside it");
#endif
    detach_from_block(ctx, o);
#ifdef MLIR_API_CHECKS
    drop_op_uses(ctx, o);  // also marks the op erased
#else
    // Without def-use tracking and regions there is nothing to drop (the
    // common case in the --from-wasm pipeline's DCE).
    if ((OPS(ctx)[o].flags & OPF_TRACKED) || OPS(ctx)[o].n_regions) drop_op_uses(ctx, o);
#endif
}

void MLIR_DetachOp(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_DetachOp");
    CHECK_OP(ctx, o, "MLIR_DetachOp");
    if (!o) return;
    detach_from_block(ctx, o);
}

MLIR_RegionHandle MLIR_TakeOpRegion(MLIR_Context *ctx, MLIR_OpHandle oh, size_t idx) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_TakeOpRegion");
    CHECK_NOT_ERASED(ctx, o, "MLIR_TakeOpRegion");
    if (!o || idx >= OPS(ctx)[o].n_regions) return MLIR_INVALID_HANDLE;
    // Replace it with a fresh empty region; the taken one is detached until
    // it is passed to MLIR_CreateOp.
    MLIR_RegionHandle fresh_h = MLIR_CreateRegion(ctx);
    uint32_t fresh = REGION_IDX(ctx, fresh_h, "MLIR_TakeOpRegion");
    REGIONS(ctx)[fresh].parent_op = o;
    uint32_t slot = OPS(ctx)[o].first_extra + (uint32_t)idx;
    uint32_t taken = EXTRA(ctx)[slot];
    EXTRA(ctx)[slot] = fresh;
    if (taken) REGIONS(ctx)[taken].parent_op = 0;
    return REGION_H(taken);
}

MLIR_BlockHandle MLIR_GetOpParentBlock(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpParentBlock");
    CHECK_NOT_ERASED(ctx, o, "MLIR_GetOpParentBlock");
    return BLOCK_H(OPS(ctx)[o].parent_block);
}

MLIR_RegionHandle MLIR_GetBlockParentRegion(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockParentRegion");
    return REGION_H(BLOCKS(ctx)[b].parent_region);
}

void MLIR_MoveOpToBlockEnd(MLIR_Context *ctx, MLIR_OpHandle oh, MLIR_BlockHandle dest) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_MoveOpToBlockEnd");
    uint32_t b = BLOCK_IDX(ctx, dest, "MLIR_MoveOpToBlockEnd");
    CHECK_NOT_ERASED(ctx, o, "MLIR_MoveOpToBlockEnd");
    if (o == 0 || b == 0) return;
    detach_from_block(ctx, o);
    block_insert_op(ctx, b, o, BLOCKS(ctx)[b].n_ops);
}

// Removes block `b` from its parent region's block list, if it has one.
static void detach_from_region(MLIR_Context *ctx, uint32_t b) {
    uint32_t r = BLOCKS(ctx)[b].parent_region;
    if (r) {
        IR_Region *reg = &REGIONS(ctx)[r];
        u32_remove(reg->blocks, &reg->n_blocks, b);
    }
    BLOCKS(ctx)[b].parent_region = 0;
}

void MLIR_MoveBlockToRegionEnd(MLIR_Context *ctx, MLIR_BlockHandle bh, MLIR_RegionHandle dest) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_MoveBlockToRegionEnd");
    uint32_t r = REGION_IDX(ctx, dest, "MLIR_MoveBlockToRegionEnd");
    if (b == 0 || r == 0) return;
    detach_from_region(ctx, b);
    MLIR_AppendRegionBlock(ctx, dest, bh);
}

// `idx` indexes the op's combined operand sequence (as upstream's
// Operation::setOperand): the regular operands, then the successor operands
// segment by segment. MLIR_GetValueUseOwner reports the same indices.
void MLIR_SetOpOperand(MLIR_Context *ctx, MLIR_OpHandle oh, size_t idx, MLIR_ValueHandle value) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_SetOpOperand");
    uint32_t v = VAL_IDX(ctx, value, "MLIR_SetOpOperand");
    CHECK_NOT_ERASED(ctx, o, "MLIR_SetOpOperand");
    if (!o) return;
    IR_Op *op = &OPS(ctx)[o];
    // idx out of range: silently no-op (as before).
    if (idx >= (size_t)op->n_operands + op->n_succ_operands) return;
    operand_set(ctx, o, op->first_operand + (uint32_t)idx, v);
}

void MLIR_SetOpSuccessor(MLIR_Context *ctx, MLIR_OpHandle oh, size_t succ_idx, MLIR_BlockHandle block) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_SetOpSuccessor");
    uint32_t b = BLOCK_IDX(ctx, block, "MLIR_SetOpSuccessor");
    CHECK_NOT_ERASED(ctx, o, "MLIR_SetOpSuccessor");
    if (!o || succ_idx >= OPS(ctx)[o].n_successors) return;
    op_succ_pair(ctx, o, succ_idx)[0] = b;
}

// Gives successor `s` of op `o` room for `new_n` operands. If the count
// changes, the op's operands move to the end of the operand vector (each
// keeping its place in its value's use list); the first min(old, new)
// operands of the segment keep their values and the added ones are empty.
static void op_resize_succ_segment(MLIR_Context *ctx, uint32_t o, size_t s, uint32_t new_n) {
    uint32_t old_n = op_succ_pair(ctx, o, s)[1];
    if (old_n == new_n) return;
    uint32_t base = op_succ_operand_base(ctx, o, s);
    uint32_t old_first = OPS(ctx)[o].first_operand;
    uint32_t old_total = OPS(ctx)[o].n_operands + OPS(ctx)[o].n_succ_operands;
    uint32_t new_total = old_total - old_n + new_n;
    uint32_t first = vec_push(&ctx->opnds, new_total);
    GEN_SAVE(ctx, opnds);
    uint32_t keep = old_n < new_n ? old_n : new_n;
    for (uint32_t i = 0; i < base + keep; i++) operand_move(ctx, old_first + i, first + i);
    // Dropped operands (shrinking) are unlinked.
    for (uint32_t k = keep; k < old_n; k++) {
        uint32_t u = old_first + base + k;
        if (operand_linked(ctx, o, OPNDS(ctx)[u].value)) use_unlink(ctx, u);
        OPNDS(ctx)[u].value = 0;
    }
    for (uint32_t k = keep; k < new_n; k++) {
        IR_Operand *rec = &OPNDS(ctx)[first + base + k];
        rec->value = 0;
        rec->owner = o;
        rec->next = rec->prev = 0;
    }
    for (uint32_t i = base + old_n; i < old_total; i++)
        operand_move(ctx, old_first + i, first + i - old_n + new_n);
    GEN_CHECK(ctx, opnds, "op_resize_succ_segment");
    IR_Op *op = &OPS(ctx)[o];
    op->first_operand = first;
    op->n_succ_operands = op->n_succ_operands - old_n + new_n;
    op_succ_pair(ctx, o, s)[1] = new_n;
}

void MLIR_SetOpSuccessorOperands(MLIR_Context *ctx, MLIR_OpHandle oh, size_t succ_idx,
                                 const MLIR_ValueHandle *values, size_t n) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_SetOpSuccessorOperands");
    CHECK_NOT_ERASED(ctx, o, "MLIR_SetOpSuccessorOperands");
    if (!o || succ_idx >= OPS(ctx)[o].n_successors) return;
    // Drop the existing uses of this segment, then link the new values in
    // order (as before #354).
    uint32_t old_n = op_succ_pair(ctx, o, succ_idx)[1];
    uint32_t first = OPS(ctx)[o].first_operand + op_succ_operand_base(ctx, o, succ_idx);
    for (uint32_t k = 0; k < old_n; k++) operand_set(ctx, o, first + k, 0);
    op_resize_succ_segment(ctx, o, succ_idx, (uint32_t)n);
    first = OPS(ctx)[o].first_operand + op_succ_operand_base(ctx, o, succ_idx);
    for (size_t k = 0; k < n; k++)
        operand_set(ctx, o, first + (uint32_t)k,
                    values ? VAL_IDX(ctx, values[k], "MLIR_SetOpSuccessorOperands") : 0);
}

void MLIR_AppendOpSuccessorOperand(MLIR_Context *ctx, MLIR_OpHandle oh, size_t succ_idx,
                                   MLIR_ValueHandle value) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_AppendOpSuccessorOperand");
    uint32_t v = VAL_IDX(ctx, value, "MLIR_AppendOpSuccessorOperand");
    CHECK_NOT_ERASED(ctx, o, "MLIR_AppendOpSuccessorOperand");
    if (!o || succ_idx >= OPS(ctx)[o].n_successors) return;
    uint32_t n = op_succ_pair(ctx, o, succ_idx)[1];
    op_resize_succ_segment(ctx, o, succ_idx, n + 1);
    uint32_t u = OPS(ctx)[o].first_operand + op_succ_operand_base(ctx, o, succ_idx) + n;
    operand_set(ctx, o, u, v);
}

void MLIR_EraseBlock(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_EraseBlock");
    if (!b) return;
#ifdef MLIR_API_CHECKS
    // No branch outside the block may still target it (mlir_api.h). The
    // native backend has no predecessor lists, so scan the region.
    {
        uint32_t r = BLOCKS(ctx)[b].parent_region;
        IR_Region *reg = r ? &REGIONS(ctx)[r] : NULL;
        for (uint32_t bi = 0; reg && bi < reg->n_blocks; bi++) {
            IR_Block *pb = &BLOCKS(ctx)[reg->blocks[bi]];
            if (reg->blocks[bi] == b || pb->n_ops == 0) continue;
            uint32_t term = pb->ops[pb->n_ops - 1];
            for (uint32_t k = 0; k < OPS(ctx)[term].n_successors; k++)
                API_CHECK(op_succ_pair(ctx, term, k)[0] != b, "MLIR_EraseBlock",
                          "the block is still a successor of a block outside it");
        }
    }
#endif
    detach_from_region(ctx, b);
}

// Inserts block `b` into region `r` at position `at` (clamped).
static void region_insert_block(MLIR_Context *ctx, uint32_t r, uint32_t b, uint32_t at) {
    IR_Region *reg = &REGIONS(ctx)[r];
    if (at > reg->n_blocks) at = reg->n_blocks;
    reg->blocks = u32_reserve(ctx->arena, reg->blocks, reg->n_blocks, &reg->cap_blocks,
                              (size_t)reg->n_blocks + 1);
    if (at < reg->n_blocks)
        memmove(reg->blocks + at + 1, reg->blocks + at, (reg->n_blocks - at) * sizeof(uint32_t));
    reg->blocks[at] = b;
    reg->n_blocks++;
    BLOCKS(ctx)[b].parent_region = r;
}

void MLIR_InsertRegionBlockAfter(MLIR_Context *ctx, MLIR_RegionHandle region,
                                 MLIR_BlockHandle block, MLIR_BlockHandle after) {
    uint32_t r = REGION_IDX(ctx, region, "MLIR_InsertRegionBlockAfter");
    uint32_t b = BLOCK_IDX(ctx, block, "MLIR_InsertRegionBlockAfter");
    uint32_t a = BLOCK_IDX(ctx, after, "MLIR_InsertRegionBlockAfter");
    if (r == 0 || b == 0) return;
    detach_from_region(ctx, b);
    // after == MLIR_INVALID_HANDLE means insert at the front; an `after`
    // that is not in the region means append.
    IR_Region *reg = &REGIONS(ctx)[r];
    uint32_t at = 0;
    if (a != 0) {
        at = reg->n_blocks;
        for (uint32_t i = 0; i < reg->n_blocks; i++)
            if (reg->blocks[i] == a) { at = i + 1; break; }
    }
    region_insert_block(ctx, r, b, at);
}

void MLIR_InsertRegionBlockBefore(MLIR_Context *ctx, MLIR_RegionHandle region,
                                  MLIR_BlockHandle block, MLIR_BlockHandle before) {
    uint32_t r = REGION_IDX(ctx, region, "MLIR_InsertRegionBlockBefore");
    uint32_t b = BLOCK_IDX(ctx, block, "MLIR_InsertRegionBlockBefore");
    uint32_t bf = BLOCK_IDX(ctx, before, "MLIR_InsertRegionBlockBefore");
    if (r == 0 || b == 0) return;
    detach_from_region(ctx, b);
    IR_Region *reg = &REGIONS(ctx)[r];
    uint32_t at = reg->n_blocks;  // append if not found / invalid
    if (bf != 0) {
        for (uint32_t i = 0; i < reg->n_blocks; i++)
            if (reg->blocks[i] == bf) { at = i; break; }
    }
    region_insert_block(ctx, r, b, at);
}

MLIR_OpHandle MLIR_GetBlockTerminator(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockTerminator");
    IR_Block *blk = &BLOCKS(ctx)[b];
    return blk->n_ops ? OP_H(blk->ops[blk->n_ops - 1]) : MLIR_INVALID_HANDLE;
}

MLIR_OpHandle MLIR_GetBlockParentOp(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockParentOp");
    uint32_t r = BLOCKS(ctx)[b].parent_region;
    return OP_H(REGIONS(ctx)[r].parent_op);
}

bool MLIR_IsBlockEntry(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_IsBlockEntry");
    if (!b) return false;
    uint32_t r = BLOCKS(ctx)[b].parent_region;
    return r && REGIONS(ctx)[r].n_blocks > 0 && REGIONS(ctx)[r].blocks[0] == b;
}

// Predecessors: walks the blocks of the parent region; each terminator
// successor slot equal to `b` is one (predecessor block, successor slot)
// entry. Returns the number of entries; stops at entry `want` (setting
// *out_block and *out_succ) if `want` is in range.
static size_t native_preds(MLIR_Context *ctx, uint32_t b, size_t want,
                           uint32_t *out_block, size_t *out_succ) {
    if (!b) return 0;
    uint32_t r = BLOCKS(ctx)[b].parent_region;
    if (!r) return 0;
    IR_Region *reg = &REGIONS(ctx)[r];
    size_t n = 0;
    for (uint32_t i = 0; i < reg->n_blocks; i++) {
        IR_Block *pb = &BLOCKS(ctx)[reg->blocks[i]];
        if (pb->n_ops == 0) continue;
        uint32_t term = pb->ops[pb->n_ops - 1];
        uint32_t ns = OPS(ctx)[term].n_successors;
        for (uint32_t s = 0; s < ns; s++) {
            if (op_succ_pair(ctx, term, s)[0] != b) continue;
            if (n == want) {
                *out_block = reg->blocks[i];
                *out_succ = s;
                return n;
            }
            n++;
        }
    }
    return n;
}

size_t MLIR_GetBlockNumPredecessors(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockNumPredecessors");
    uint32_t pb = 0;
    size_t s = 0;
    return native_preds(ctx, b, SIZE_MAX, &pb, &s);
}

MLIR_BlockHandle MLIR_GetBlockPredecessor(MLIR_Context *ctx, MLIR_BlockHandle bh, size_t idx,
                                          size_t *out_succ_idx) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockPredecessor");
    uint32_t pb = 0;
    size_t s = SIZE_MAX;
    native_preds(ctx, b, idx, &pb, &s);
    if (out_succ_idx) *out_succ_idx = pb ? s : SIZE_MAX;
    return BLOCK_H(pb);
}

MLIR_ValueHandle MLIR_AddBlockArgument(MLIR_Context *ctx, MLIR_BlockHandle bh,
                                       MLIR_TypeHandle type, MLIR_LocationHandle loc) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_AddBlockArgument");
    if (!b) return MLIR_INVALID_HANDLE;
    uint32_t v = value_create(ctx, b, BLOCKS(ctx)[b].n_args, type, loc, (string){0});
    VALS(ctx)[v].index |= VAL_BLOCK_ARG;
    IR_Block *blk = &BLOCKS(ctx)[b];
    blk->args = u32_reserve(ctx->arena, blk->args, blk->n_args, &blk->cap_args, (size_t)blk->n_args + 1);
    blk->args[blk->n_args++] = v;
    return VAL_H(v);
}

void MLIR_EraseBlockArguments(MLIR_Context *ctx, MLIR_BlockHandle bh, size_t start, size_t count) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_EraseBlockArguments");
    if (!b || count == 0) return;
    IR_Block *blk = &BLOCKS(ctx)[b];
    if (start >= blk->n_args) return;
    if (start + count > blk->n_args) count = blk->n_args - start;
    for (size_t i = start; i + count < blk->n_args; i++)
        blk->args[i] = blk->args[i + count];
    blk->n_args -= (uint32_t)count;
    // Renumber the remaining arguments.
    for (size_t i = start; i < blk->n_args; i++) {
        IR_Value *v = &VALS(ctx)[blk->args[i]];
        v->index = (v->index & ~VAL_INDEX_MASK) | (uint32_t)i;
    }
}

MLIR_BlockHandle MLIR_GetValueParentBlock(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_GetValueParentBlock");
    if (!v) return MLIR_INVALID_HANDLE;
    IR_Value *p = &VALS(ctx)[v];
    if (p->index & VAL_BLOCK_ARG) return BLOCK_H(p->def);
    return BLOCK_H(OPS(ctx)[p->def].parent_block);
}

size_t MLIR_GetValueNumUses(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_GetValueNumUses");
    return VALS(ctx)[v].n_uses;
}

// The owner of use `idx` (newest first); *out_operand_idx is the operand's
// index in the owner's combined operand sequence (see MLIR_SetOpOperand).
MLIR_OpHandle MLIR_GetValueUseOwner(MLIR_Context *ctx, MLIR_ValueHandle vh,
                                    size_t idx, size_t *out_operand_idx) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_GetValueUseOwner");
    if (out_operand_idx) *out_operand_idx = SIZE_MAX;
    uint32_t u = VALS(ctx)[v].first_use;
    for (size_t n = 0; u && n < idx; n++) u = OPNDS(ctx)[u].next;
    if (!u) return MLIR_INVALID_HANDLE;
    uint32_t owner = OPNDS(ctx)[u].owner;
    if (out_operand_idx) *out_operand_idx = u - OPS(ctx)[owner].first_operand;
    return OP_H(owner);
}

void MLIR_SpliceBlockOps(MLIR_Context *ctx, MLIR_BlockHandle dst, MLIR_BlockHandle src) {
    uint32_t d = BLOCK_IDX(ctx, dst, "MLIR_SpliceBlockOps");
    uint32_t s = BLOCK_IDX(ctx, src, "MLIR_SpliceBlockOps");
    if (!d || !s || BLOCKS(ctx)[s].n_ops == 0) return;
    IR_Block *db = &BLOCKS(ctx)[d];
    IR_Block *sb = &BLOCKS(ctx)[s];
    size_t new_n = (size_t)db->n_ops + sb->n_ops;
    db->ops = u32_reserve(ctx->arena, db->ops, db->n_ops, &db->cap_ops, new_n);
    memcpy(db->ops + db->n_ops, sb->ops, sb->n_ops * sizeof(uint32_t));
    for (uint32_t i = 0; i < sb->n_ops; i++) OPS(ctx)[sb->ops[i]].parent_block = d;
    db->n_ops = (uint32_t)new_n;
    sb->ops = NULL;
    sb->n_ops = 0;
    sb->cap_ops = 0;
}

// Native cf->scf lift: dispatch to the agnostic C port. The port handles
// the patterns it understands and returns false otherwise; for native we
// do not have an upstream fallback so callers (wasm lowering) will reject
// any leftover branch ops at the wasmssa-lower stage.
#include "mlir_lift_cf_to_scf.h"
bool MLIR_LiftCfToScf(MLIR_Context *ctx, MLIR_OpHandle module,
                      const MLIR_LiftCfToScfOptions *options) {
    return MLIR_LiftCfToScfNative(ctx, module, options);
}

// The lowering / LLVM-IR-translation / wasm-translation entry points
// declared in mlir_passes.h are implemented in dedicated agnostic
// translation units (mlir_lower_to_llvm.c, mlir_translate_to_llvm_ir.c,
// mlir_translate_to_wasm.c). They are linked into hosted builds
// alongside this file. There is no parser-only "native" build, so we
// no longer need fallback stubs here.

#ifdef __cplusplus
}
#endif

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
#include "mlir_generic_printer.h"
#include <platform/platform.h>
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

#if defined(__GNUC__) || defined(__clang__)
#define MLIR_NAT_NOINLINE __attribute__((noinline))
#define MLIR_NAT_LIKELY(x) __builtin_expect(!!(x), 1)
#define MLIR_NAT_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define MLIR_NAT_NOINLINE
#define MLIR_NAT_LIKELY(x) (x)
#define MLIR_NAT_UNLIKELY(x) (x)
#endif

// Internal concrete IR definitions

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
//     count) pair per successor; separately, an op's attributes as (key,
//     attribute index) pairs, and the explicit result types (type indices)
//     of the rare op whose results do not carry them.
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

// IR_Operand.owner packs the 26-bit op index (up to 67M ops) in bits 0..25
// and a 6-bit inline operand index (0..62; 63 = escape to OPS[o].first_operand)
// in bits 26..31 so MLIR_GetUseOperandIndex avoids touching OPS(ctx).
#define OPND_OWNER_MASK 0x03ffffffu
#define OPND_IDX_SHIFT  26u
#define OPND_IDX_ESCAPE 63u

static inline uint32_t opnd_pack_owner(uint32_t op, uint32_t idx) {
    API_CHECK(op <= OPND_OWNER_MASK, "MLIR", "op index exceeds 26-bit operand owner field");
    uint32_t tag = (idx < OPND_IDX_ESCAPE) ? idx : OPND_IDX_ESCAPE;
    return op | (tag << OPND_IDX_SHIFT);
}

typedef struct IR_Operand {
    uint32_t value;  // value index, 0 = none
    uint32_t owner;  // op index (bits 0..25) | operand_idx (bits 26..31)
    uint32_t next;   // next use of `value` (operand index), 0 = end
    uint32_t prev;   // previous use, 0 = this is the value's first use
} IR_Operand;

// IR_Value.index: the result or argument index, plus these flags.
#define VAL_BLOCK_ARG  0x80000000u  // a block argument (else an op result)
#define VAL_TRACKED    0x40000000u  // has a use list (see value_create)
#define VAL_HAS_LOC    0x20000000u  // has explicit location in ctx->vloc_map
#define VAL_HAS_NAME   0x10000000u  // has register name in ctx->vname_map
#define VAL_INDEX_MASK 0x0fffffffu

// 20-byte compact value record: op-result locations come from the defining op,
// and rare register names / block-arg locations live in lazy side tables.
typedef struct IR_Value {
    uint32_t type;       // type index, 0 = none
    uint32_t def;        // op index (result) or block index (block argument)
    uint32_t index;      // result/argument index | VAL_* flags
    uint32_t first_use;  // operand index, 0 = no uses
    uint32_t n_uses;
} IR_Value;

// Lazy open-addressing u32 -> u32 side table (0 = empty key).
typedef struct IR_U32Map {
    uint32_t *keys;
    uint32_t *vals;
    uint32_t cap;
    uint32_t n;
} IR_U32Map;

// IR_Op.flags
#define OPF_TRACKED      1u  // operands were linked into use lists at creation
#define OPF_HAS_EXTRA    2u  // has regions or successors (in op_extra_map or first_result)
#define OPF_HAS_AUX      4u  // has rare printer/parser aux (in ctx->op_aux_map)
#define OPF_INLINE_EXTRA 8u  // n_results == 0: op_extra index stored inline in first_result

// 32-byte compact op record (exact half cache line — 2 ops per 64-byte line).
typedef struct IR_Op {
    uint16_t op_type;       // MLIR_OpType
    uint8_t  flags;         // OPF_*
    uint8_t  n_attributes;  // attribute count (0..255)
    uint16_t n_operands;    // regular operands
    uint16_t n_results;
    uint32_t parent_block;  // block index, 0 = not in a block
    uint32_t block_pos;     // index in parent_block->ops
    uint32_t first_operand; // opnds index
    uint32_t first_result;  // vals index
    uint32_t first_attr;    // extra index: (key, attribute index) pairs
    uint32_t location;      // location index, 0 = none
} IR_Op;

// Rare op metadata for ops with regions or successors (~5% of ops).
typedef struct IR_OpExtra {
    uint32_t first_extra;     // extra index: regions, then (block, count, pred) triples
    uint16_t n_regions;
    uint16_t n_successors;
    uint16_t n_succ_operands; // all successor operands, after the regular ones
    uint16_t pad;
} IR_OpExtra;

// Rarely-used op metadata consulted only by the classic printer and the
// parser roundtrip (trailing comments, the unnumbered location def, the
// recorded source line), plus the result types of an op whose results do not
// carry them (MLIR_CreateOp with n_result_types != n_results).
typedef struct IR_OpAux {
    uint32_t unnumbered_loc_def;  // location index
    string trailing_comment;
    int32_t source_line_start;
    uint32_t first_result_type;  // extra index (type indices)
    uint32_t n_result_types;
    bool explicit_result_types;
} IR_OpAux;

typedef struct IR_PredEdge {
    uint32_t owner;        // terminator op index
    uint32_t succ_idx;     // successor slot index in owner
    uint32_t target_block; // target block index, 0 = unlinked
    uint32_t next;         // next pred edge of target_block, 0 = end
    uint32_t prev;         // prev pred edge of target_block, 0 = head
} IR_PredEdge;

typedef struct IR_Block {
    uint32_t *ops;
    uint32_t *args;
    uint32_t n_ops, cap_ops;
    uint32_t n_args, cap_args;
    uint32_t parent_region;  // region index, 0 = detached
    uint32_t region_pos;     // index in parent_region->blocks
    uint32_t first_pred;     // pred_edges index, 0 = none
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

// --- Types (#349) -------------------------------------------------------------
//
// Types are hash-consed into the context's `types` vector, and a type handle
// is the index of its record, so equal types have equal handles. A record
// holds everything the getters return (kind, width, element type, element
// count), so each of them is one array read; the variable-length parts
// (function inputs and results, struct fields, shape dimensions) are words
// of the `tyx` vector. Every child type is interned before its parent, so a
// type hashes over its kind, its scalar fields and its children's indices.
// Identified LLVM structs are interned by name instead, and their body (the
// fields) is set once, after creation. Types are never removed.

#define TYF_VAR_ARG    1u  // LLVM function type: variadic
#define TYF_IDENTIFIED 2u  // interned by name: identified LLVM struct, opaque type
#define TYF_BODY_SET   4u  // identified LLVM struct: the body has been set
#define TYF_TT_PTR     8u  // OPAQUE: a !tt.ptr<pointee[, address space]>
#define TYF_HAS_AS     16u // !tt.ptr: the address space is printed

typedef struct IR_Type {
    uint8_t kind;     // MLIR_TypeKind
    uint8_t flags;    // TYF_*
    uint16_t pad;
    uint32_t width;   // integer and float types only (MLIR_GetTypeWidth)
    uint32_t elem;    // element type of an LLVM array, vector, tensor, memref
    uint32_t aux;     // pointer address space; function: number of results;
                      // identified struct: name (tnames index)
    uint32_t first;   // tyx index of the child words
    uint32_t n;       // number of child words: function inputs then results;
                      // LLVM function result then inputs; struct fields;
                      // shape dimensions (two words each); tt.ptr pointee
    uint64_t count;   // LLVM array or vector: number of elements
} IR_Type;

// --- Attributes (#350) ----------------------------------------------------------
//
// Attribute values are hash-consed into the context's `attrs` vector, like
// types: an attribute handle is the record's index, so equal values have
// equal handles. A record holds the scalar payload; arrays, dictionaries
// and dense arrays keep their elements as words of `ax` (attribute indices,
// (key, attribute index) pairs, or the dense values, 32 bits per i32 and 64
// per i64 element), and a string's bytes are in the persistent arena. Floats
// hash and compare by bit pattern. Distinct attributes are never looked up:
// each gets a new record.

typedef enum {
    AK_NONE = 0,
    AK_INTEGER,     // bits: the value, type: its type
    AK_FLOAT,       // bits: the IEEE double's bit pattern, type: its type
    AK_STRING,      // bits: pointer to the bytes, n: the length (symbol
                    // refs `@name` and LLVM linkages `#llvm.linkage<...>` too)
    AK_BOOL,        // bits: 0 or 1
    AK_ARRAY,       // n attribute indices from ax[first]
    AK_DICT,        // n (key, attribute index) pairs from ax[first]
    AK_TYPE,        // type: the payload
    AK_DENSE_I32,   // n values, one word each, from ax[first]
    AK_DENSE_I64,   // n values, two words each (low, high), from ax[first]
    AK_DISTINCT,    // type: the referenced attribute's index; bits: number
    AK_FASTMATH,    // bits: MLIR_FastMath bitmask
    AK_SPLAT        // type: vector/shaped type index; bits: scalar attribute index
} AttrKindInternal;

typedef struct IR_Attr {
    uint8_t kind;     // AttrKindInternal
    uint8_t pad8;
    uint16_t pad16;
    uint32_t type;
    uint32_t first;
    uint32_t n;
    uint64_t bits;
    uint32_t text;    // atext index of the printed form (dense), 0 = none yet
    uint32_t pad;
} IR_Attr;

// --- Locations (#350) ----------------------------------------------------------
//
// A location is an index into the context's `locs` vector; locations are
// not uniqued. `str` is a loc_strings index: the file name (FILE), the name
// (NAME) or the parser's original text (UNKNOWN), 0 = none. Element 1 is the
// shared unknown location without text.

typedef struct IR_Loc {
    uint32_t kind;    // MLIR_LocationKind
    uint32_t str;
    int32_t line;     // FILE: line; REF: the reference id
    int32_t col;
    uint32_t text;    // loc_strings index of the `loc(...)` text, 0 = not made yet
} IR_Loc;

// The context. Everything that is not IR lives here; there is no
// process-wide state. The context owns `persistent` (this struct, the names
// of op types and identified structs) and the type vectors, so types outlive
// every IR arena.
struct MLIR_Context {
    Arena *arena;          // current IR arena (owned by the caller)
    Arena *persistent;     // owned: this struct, types, the interning tables
    bool no_def_use_tracking;
    uint32_t salt;         // MLIR_API_CHECKS handle salt (see h_make)
    // The IR vectors (see above), owned by the context.
    IR_Vec ops, vals, opnds, extra, blocks, regions, op_extra, op_aux, names;
    // Lazy side tables for rare per-op / per-value metadata.
    IR_U32Map op_extra_map, op_aux_map, vloc_map, vname_map;
    // The uniqued types (see "Types" below): the type records, their child
    // words, the names of identified structs and the hash table that
    // uniques them.
    IR_Vec types, tyx, tnames;
    uint32_t *tmap;        // open addressing, type indices, 0 = empty
    uint32_t tmap_cap;     // a power of two
    uint32_t tmap_n;
    uint32_t int_types[65];  // iN for N <= 64, 0 = not created yet
    uint32_t ptr_type;       // !llvm.ptr, 0 = not created yet
    // The uniqued attributes (see "Attributes" below): the records, their
    // child words, the cached printed forms and the hash table.
    IR_Vec attrs, ax, atext;
    uint32_t *amap;
    uint32_t amap_cap;
    uint32_t amap_n;
    // Locations: the records, and the interned file names, names and
    // parser texts they refer to.
    IR_Vec locs;
    AttrKeyTable loc_strings;
    // Predecessor edge list records (indexed by 1-based MLIR_PredHandle).
    IR_Vec preds;
    // Diagnostic handler; NULL means the default (print to stderr).
    MLIR_DiagnosticHandler diag_fn;
    void *diag_user;
    // Op types: the names of the static and dynamic types and the reverse
    // map (MLIR_RegisterOpType).
    OpTypeTable op_types;
    // Attribute keys (MLIR_InternAttrKey).
    AttrKeyTable keys;
};

#define OPS(c)     ((IR_Op *)(c)->ops.data)
#define VALS(c)    ((IR_Value *)(c)->vals.data)
#define OPNDS(c)   ((IR_Operand *)(c)->opnds.data)
#define EXTRA(c)   ((uint32_t *)(c)->extra.data)
#define BLOCKS(c)  ((IR_Block *)(c)->blocks.data)
#define REGIONS(c) ((IR_Region *)(c)->regions.data)
#define OPEXTRA(c) ((IR_OpExtra *)(c)->op_extra.data)
#define OPAUX(c)   ((IR_OpAux *)(c)->op_aux.data)
#define NAMES(c)   ((string *)(c)->names.data)
#define TYPES(c)   ((IR_Type *)(c)->types.data)
#define TYX(c)     ((uint32_t *)(c)->tyx.data)
#define TNAMES(c)  ((string *)(c)->tnames.data)
#define ATTRS(c)   ((IR_Attr *)(c)->attrs.data)
#define AX(c)      ((uint32_t *)(c)->ax.data)
#define ATEXT(c)   ((string *)(c)->atext.data)
#define LOCS(c)    ((IR_Loc *)(c)->locs.data)
#define PREDS(c)   ((IR_PredEdge *)(c)->preds.data)

static void pred_push(MLIR_Context *c, uint32_t e) {
    IR_PredEdge *pe = &PREDS(c)[e];
    uint32_t b = pe->target_block;
    if (!b) return;
    uint32_t head = BLOCKS(c)[b].first_pred;
    pe->prev = 0;
    pe->next = head;
    if (head) PREDS(c)[head].prev = e;
    BLOCKS(c)[b].first_pred = e;
}

static void pred_unlink(MLIR_Context *c, uint32_t e) {
    IR_PredEdge *pe = &PREDS(c)[e];
    uint32_t b = pe->target_block;
    if (!b) return;
    if (pe->prev) PREDS(c)[pe->prev].next = pe->next;
    else BLOCKS(c)[b].first_pred = pe->next;
    if (pe->next) PREDS(c)[pe->next].prev = pe->prev;
    pe->prev = pe->next = 0;
    pe->target_block = 0;
}

// The one growth hook: makes room for `need` elements. The capacity at least
// doubles (or becomes `need`, if that is more) and uses all of the buddy
// block it gets; the elements in use are copied and the old array is freed,
// so every pointer into the vector dies.
// (A later change can reserve address space here instead, so that the
// arrays never move on hosts that have it.)
static MLIR_NAT_NOINLINE void vec_reserve(IR_Vec *v, size_t need) {
    if (need <= v->cap) return;
    size_t want = v->cap ? (size_t)v->cap * 2 : 16;
    if (want < need) want = need;
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
    if (MLIR_NAT_UNLIKELY((size_t)v->n + k > v->cap)) vec_reserve(v, (size_t)v->n + k);
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
// with its kind in the top four bits and XOR the index with a per-context
// salt in bits 20-27, so that a handle of another kind is caught, and a
// handle of another context decodes to an index out of range (certainly
// while the contexts have fewer than 2^20 elements of the kind and their
// salts differ, which the contexts' addresses decide). Release builds use
// the bare index.
#ifdef MLIR_API_CHECKS
#define HK_OP 1u
#define HK_VALUE 2u
#define HK_BLOCK 3u
#define HK_REGION 4u
#define HK_TYPE 5u
#define HK_ATTR 6u
#define HK_LOC 7u
#define H_SHIFT 28
#define H_MASK 0x0fffffffu
static uint32_t h_index(MLIR_Context *c, uintptr_t h, uint32_t kind, uint32_t n, const char *fn) {
    if (h == 0) return 0;
    if ((uint32_t)(h >> H_SHIFT) != kind) api_check_failed(fn, "handle of another kind, or not a handle");
    uint32_t i = (uint32_t)(h & H_MASK) ^ c->salt;
    if (i == 0 || i >= n) api_check_failed(fn, "handle out of range (from another context?)");
    return i;
}
static inline uintptr_t h_make(MLIR_Context *c, uint32_t i, uint32_t kind) {
    if (i == 0) return 0;
    if (i > H_MASK) api_check_failed("MLIR", "too many IR objects for a checked build");
    return (uintptr_t)(i ^ c->salt) | ((uintptr_t)kind << H_SHIFT);
}
#define OP_IDX(c, h, fn)     h_index((c), (h), HK_OP, (c)->ops.n, (fn))
#define VAL_IDX(c, h, fn)    h_index((c), (h), HK_VALUE, (c)->vals.n, (fn))
#define BLOCK_IDX(c, h, fn)  h_index((c), (h), HK_BLOCK, (c)->blocks.n, (fn))
#define REGION_IDX(c, h, fn) h_index((c), (h), HK_REGION, (c)->regions.n, (fn))
#define TYPE_IDX(c, h, fn)   h_index((c), (h), HK_TYPE, (c)->types.n, (fn))
#define ATTR_IDX(c, h, fn)   h_index((c), (h), HK_ATTR, (c)->attrs.n, (fn))
#define LOC_IDX(c, h, fn)    h_index((c), (h), HK_LOC, (c)->locs.n, (fn))
#define OP_H(c, i)     ((MLIR_OpHandle)h_make((c), (i), HK_OP))
#define VAL_H(c, i)    ((MLIR_ValueHandle)h_make((c), (i), HK_VALUE))
#define BLOCK_H(c, i)  ((MLIR_BlockHandle)h_make((c), (i), HK_BLOCK))
#define REGION_H(c, i) ((MLIR_RegionHandle)h_make((c), (i), HK_REGION))
#define TYPE_H(c, i)   ((MLIR_TypeHandle)h_make((c), (i), HK_TYPE))
#define ATTR_H(c, i)   ((MLIR_AttributeHandle)h_make((c), (i), HK_ATTR))
#define LOC_H(c, i)    ((MLIR_LocationHandle)h_make((c), (i), HK_LOC))
#else
#define OP_IDX(c, h, fn)     ((uint32_t)(h))
#define VAL_IDX(c, h, fn)    ((uint32_t)(h))
#define BLOCK_IDX(c, h, fn)  ((uint32_t)(h))
#define REGION_IDX(c, h, fn) ((uint32_t)(h))
#define TYPE_IDX(c, h, fn)   ((uint32_t)(h))
#define ATTR_IDX(c, h, fn)   ((uint32_t)(h))
#define LOC_IDX(c, h, fn)    ((uint32_t)(h))
#define OP_H(c, i)     ((MLIR_OpHandle)(i))
#define VAL_H(c, i)    ((MLIR_ValueHandle)(i))
#define BLOCK_H(c, i)  ((MLIR_BlockHandle)(i))
#define REGION_H(c, i) ((MLIR_RegionHandle)(i))
#define TYPE_H(c, i)   ((MLIR_TypeHandle)(i))
#define ATTR_H(c, i)   ((MLIR_AttributeHandle)(i))
#define LOC_H(c, i)    ((MLIR_LocationHandle)(i))
#endif

// Under MLIR_API_CHECKS an erased op keeps this type, so every later use of
// its handle is caught. Without checks MLIR_EraseOp leaves the type alone.
#define OP_TYPE_ERASED_SENTINEL ((uint16_t)0xffffu)

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
// A typed attribute getter applied to the wrong kind (`a` is the record).
#define CHECK_ATTR_KIND(ah, ok, fn, what) do { \
        API_CHECK((ah) != MLIR_INVALID_HANDLE, fn, "attribute handle is NULL"); \
        API_CHECK(ok, fn, what); \
    } while (0)

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
    if (operand_linked(c, o[to].owner & OPND_OWNER_MASK, o[to].value)) {
        if (o[to].prev) o[o[to].prev].next = to;
        else            VALS(c)[o[to].value].first_use = to;
        if (o[to].next) o[o[to].next].prev = to;
    }
    o[from].value = 0;
    o[from].next = o[from].prev = 0;
}

static void u32map_free(IR_U32Map *m) {
    if (m->keys) buddy_free(m->keys);
    if (m->vals) buddy_free(m->vals);
    m->keys = m->vals = NULL;
    m->cap = m->n = 0;
}

static MLIR_NAT_NOINLINE void u32map_grow(IR_U32Map *m) {
    uint32_t ncap = m->cap ? m->cap * 2 : 64;
    size_t dummy = 0;
    uint32_t *nk = (uint32_t *)buddy_alloc((size_t)ncap * sizeof(uint32_t), &dummy);
    uint32_t *nv = (uint32_t *)buddy_alloc((size_t)ncap * sizeof(uint32_t), &dummy);
    if (!nk || !nv) {
        fprintf(stderr, "MLIR: out of memory growing an IR side map to %u entries\n", ncap);
        abort();
    }
    memset(nk, 0, (size_t)ncap * sizeof(uint32_t));
    if (m->cap) {
        uint32_t mask = ncap - 1;
        for (uint32_t i = 0; i < m->cap; i++) {
            uint32_t k = m->keys[i];
            if (!k) continue;
            uint32_t s = (k * 2654435761u) & mask;
            while (nk[s]) s = (s + 1) & mask;
            nk[s] = k;
            nv[s] = m->vals[i];
        }
        buddy_free(m->keys);
        buddy_free(m->vals);
    }
    m->keys = nk;
    m->vals = nv;
    m->cap = ncap;
}

static inline uint32_t u32map_get(const IR_U32Map *m, uint32_t k) {
    if (!m->cap || !k) return 0;
    uint32_t mask = m->cap - 1;
    uint32_t s = (k * 2654435761u) & mask;
    while (m->keys[s]) {
        if (m->keys[s] == k) return m->vals[s];
        s = (s + 1) & mask;
    }
    return 0;
}

static MLIR_NAT_NOINLINE void u32map_put(IR_U32Map *m, uint32_t k, uint32_t v) {
    if (!k) return;
    if ((m->n + 1) * 2 > m->cap) u32map_grow(m);
    uint32_t mask = m->cap - 1;
    uint32_t s = (k * 2654435761u) & mask;
    while (m->keys[s] && m->keys[s] != k) s = (s + 1) & mask;
    if (!m->keys[s]) m->n++;
    m->keys[s] = k;
    m->vals[s] = v;
}

// A new value record. A value has a use list if the context tracks uses
// when it is created, or if it has a register name (as before #354).
static uint32_t value_create(MLIR_Context *c, uint32_t def, uint32_t index,
                             uint32_t type, MLIR_LocationHandle loc,
                             string register_name) {
    uint32_t v = vec_push(&c->vals, 1);
    IR_Value *p = &VALS(c)[v];
    p->type = type;
    p->def = def;
    p->index = index & VAL_INDEX_MASK;
    if (!c->no_def_use_tracking || register_name.size != 0) p->index |= VAL_TRACKED;
    p->first_use = 0;
    p->n_uses = 0;
    uint32_t loc_idx = LOC_IDX(c, loc, "MLIR_CreateValue");
    if (loc_idx != 0) {
        p->index |= VAL_HAS_LOC;
        u32map_put(&c->vloc_map, v, loc_idx);
    }
    if (register_name.size != 0) {
        uint32_t n = vec_push(&c->names, 1);
        NAMES(c)[n] = register_name;
        VALS(c)[v].index |= VAL_HAS_NAME;
        u32map_put(&c->vname_map, v, n);
    }
    return v;
}

static inline const IR_OpExtra *op_extra_get(MLIR_Context *c, uint32_t o) {
    uint8_t fl = OPS(c)[o].flags;
    if (MLIR_NAT_LIKELY(!(fl & OPF_HAS_EXTRA))) return NULL;
    if (MLIR_NAT_LIKELY(fl & OPF_INLINE_EXTRA)) return &OPEXTRA(c)[OPS(c)[o].first_result];
    return &OPEXTRA(c)[u32map_get(&c->op_extra_map, o)];
}

static inline IR_OpExtra *op_extra_mut(MLIR_Context *c, uint32_t o) {
    uint8_t fl = OPS(c)[o].flags;
    if (!(fl & OPF_HAS_EXTRA)) return NULL;
    if (fl & OPF_INLINE_EXTRA) return &OPEXTRA(c)[OPS(c)[o].first_result];
    return &OPEXTRA(c)[u32map_get(&c->op_extra_map, o)];
}

static inline uint32_t op_aux_lookup(MLIR_Context *c, uint32_t o) {
    if (MLIR_NAT_LIKELY(!(OPS(c)[o].flags & OPF_HAS_AUX))) return 0;
    return u32map_get(&c->op_aux_map, o);
}

// The op's aux record, created on first use.
static uint32_t op_aux_get(MLIR_Context *c, uint32_t o) {
    uint32_t a = op_aux_lookup(c, o);
    if (a) return a;
    a = vec_push(&c->op_aux, 1);
    IR_OpAux *p = &OPAUX(c)[a];
    p->unnumbered_loc_def = 0;
    p->trailing_comment = (string){0};
    p->source_line_start = -1;
    p->first_result_type = 0;
    p->n_result_types = 0;
    p->explicit_result_types = false;
    OPS(c)[o].flags |= OPF_HAS_AUX;
    u32map_put(&c->op_aux_map, o, a);
    return a;
}

// Grows an arena array of 32-bit indices so it holds `need` entries; the
// old array is left to the arena.
static MLIR_NAT_NOINLINE uint32_t *u32_grow(Arena *arena, uint32_t *old, uint32_t n, uint32_t *cap, size_t need) {
    size_t nc = *cap ? (size_t)*cap * 2 : 8;
    while (nc < need) nc *= 2;
    uint32_t *p = arena_new_array(arena, uint32_t, nc);
    if (n) memcpy(p, old, (size_t)n * sizeof(uint32_t));
    *cap = (uint32_t)nc;
    return p;
}

static inline uint32_t *u32_reserve(Arena *arena, uint32_t *old, uint32_t n, uint32_t *cap, size_t need) {
    if (MLIR_NAT_LIKELY(need <= *cap)) return old;
    return u32_grow(arena, old, n, cap, need);
}

// Removes `x`, which occurs at most once, from `arr[0..*n)`, keeping the
// order: a scan to it, then the tail moves down by one. (Erasing k ops of a
// block is still O(k * n), N7 in doc/mlir_api_design.md.)
static void u32_remove(uint32_t *arr, uint32_t *n, uint32_t x) {
    // Pointer loops, not memmove: corec's memmove, which the self-hosted
    // compiler uses, copies a byte at a time, and these are the fastest
    // form under both gcc and tinyC.
    uint32_t *p = arr;
    uint32_t *end = arr + *n;
    while (p < end && *p != x) p++;
    if (p == end) return;
    uint32_t *q = p + 1;
    while (q < end) { *p = *q; p++; q++; }
    *n = *n - 1;
}

// --- Type interning ------------------------------------------------------------

static inline uint32_t type_mix(uint32_t h, uint32_t w) {
    h ^= w;
    h *= 0x9E3779B1u;
    return h ^ (h >> 15);
}

// The interning hash: an identified struct's name, or a type's kind, scalar
// fields and child words.
static uint32_t type_hash(MLIR_Context *c, const IR_Type *t, const uint32_t *kids) {
    uint32_t h = 0x811C9DC5u;
    if (t->flags & TYF_IDENTIFIED) {
        string name = TNAMES(c)[t->aux];
        h = type_mix(h, t->kind);
        for (size_t i = 0; i < name.size; i++) h = type_mix(h, (uint8_t)name.str[i]);
        return h;
    }
    h = type_mix(h, (uint32_t)t->kind | ((uint32_t)t->flags << 8));
    h = type_mix(h, t->width);
    h = type_mix(h, t->elem);
    h = type_mix(h, t->aux);
    h = type_mix(h, (uint32_t)t->count);
    h = type_mix(h, (uint32_t)(t->count >> 32));
    h = type_mix(h, t->n);
    for (uint32_t i = 0; i < t->n; i++) h = type_mix(h, kids[i]);
    return h;
}

// True if record `a` (with child words `ak`) and the candidate `b` (`bk`)
// are the same type. `b`'s name, for an identified struct, is `bname`.
static bool type_same(MLIR_Context *c, const IR_Type *a, const uint32_t *ak,
                      const IR_Type *b, const uint32_t *bk, string bname) {
    if (a->kind != b->kind) return false;
    if ((a->flags | b->flags) & TYF_IDENTIFIED) {
        if (!((a->flags & b->flags) & TYF_IDENTIFIED)) return false;
        string an = TNAMES(c)[a->aux];
        return an.size == bname.size && memcmp(an.str, bname.str, an.size) == 0;
    }
    if (a->flags != b->flags || a->width != b->width || a->elem != b->elem ||
        a->aux != b->aux || a->count != b->count || a->n != b->n) return false;
    for (uint32_t i = 0; i < a->n; i++)
        if (ak[i] != bk[i]) return false;
    return true;
}

// Inserts type `t` (already in `types`) into the hash table.
static void tmap_insert(MLIR_Context *c, uint32_t t, uint32_t h) {
    uint32_t mask = c->tmap_cap - 1;
    uint32_t i = h & mask;
    while (c->tmap[i] != 0) i = (i + 1) & mask;
    c->tmap[i] = t;
    c->tmap_n++;
}

static void tmap_alloc(MLIR_Context *c, uint32_t cap) {
    size_t actual = 0;
    c->tmap = (uint32_t *)buddy_alloc((size_t)cap * sizeof(uint32_t), &actual);
    if (!c->tmap) {
        fprintf(stderr, "MLIR: out of memory growing the type table\n");
        abort();
    }
    memset(c->tmap, 0, (size_t)cap * sizeof(uint32_t));
    c->tmap_cap = cap;
    c->tmap_n = 0;
}

// Doubles the hash table and rehashes every type.
static void tmap_grow(MLIR_Context *c) {
    uint32_t *old = c->tmap;
    tmap_alloc(c, c->tmap_cap * 2);
    for (uint32_t t = 1; t < c->types.n; t++) {
        IR_Type *r = &TYPES(c)[t];
        tmap_insert(c, t, type_hash(c, r, TYX(c) + r->first));
    }
    buddy_free(old);
}

// Interns the candidate `t`. Its child words, if any, are the last `t.n`
// words of `tyx`, pushed by the caller from `t.first` on: on a hit they are
// popped again and the existing type's index is returned, so looking up an
// existing type allocates nothing; on a miss `t` becomes a new type. `name`
// is the name of an identified struct (copied on a miss), else empty.
static uint32_t type_intern(MLIR_Context *c, IR_Type t, string name) {
    uint32_t h;
    if (t.flags & TYF_IDENTIFIED) {
        // Hash the candidate's name, which is not in tnames yet.
        h = type_mix(0x811C9DC5u, t.kind);
        for (size_t i = 0; i < name.size; i++) h = type_mix(h, (uint8_t)name.str[i]);
    } else {
        h = type_hash(c, &t, TYX(c) + t.first);
    }
    uint32_t mask = c->tmap_cap - 1;
    for (uint32_t i = h & mask; c->tmap[i] != 0; i = (i + 1) & mask) {
        uint32_t e = c->tmap[i];
        IR_Type *r = &TYPES(c)[e];
        if (type_same(c, r, TYX(c) + r->first, &t, TYX(c) + t.first, name)) {
            if (t.n) c->tyx.n = t.first;
            return e;
        }
    }
    if (t.flags & TYF_IDENTIFIED) {
        string owned = name;
        if (name.size > 0) {
            char *p = arena_new_array(c->persistent, char, name.size);
            memcpy(p, name.str, name.size);
            owned.str = p;
        }
        t.aux = vec_push(&c->tnames, 1);
        TNAMES(c)[t.aux] = owned;
    }
    // Grow first: the rehash inserts every type already in `types`.
    if ((c->tmap_n + 1) * 2 > c->tmap_cap) tmap_grow(c);
    uint32_t idx = vec_push(&c->types, 1);
    TYPES(c)[idx] = t;
    tmap_insert(c, idx, h);
    return idx;
}

// Pushes `n` type handles as child words; returns the index of the first.
static uint32_t tyx_push_types(MLIR_Context *c, const MLIR_TypeHandle *ts, size_t n, const char *fn) {
    uint32_t first = vec_push(&c->tyx, (uint32_t)n);
    for (size_t i = 0; i < n; i++) TYX(c)[first + i] = TYPE_IDX(c, ts[i], fn);
    (void)fn;
    return first;
}

// An empty candidate of kind `k`.
static IR_Type type_make(MLIR_TypeKind k) {
    IR_Type t;
    memset(&t, 0, sizeof t);
    t.kind = (uint8_t)k;
    return t;
}

// --- Attribute interning ------------------------------------------------------
//
// Attributes are created for most ops, so this path is written for tinyC,
// which compiles it when self-hosting: no struct is passed by value or
// cleared with memset, and the mixing is spelled out instead of calling a
// helper per word.

#define ATTR_MIX(h, w) do { (h) ^= (uint32_t)(w); (h) *= 0x9E3779B1u; (h) ^= (h) >> 15; } while (0)

// The number of `ax` words of an attribute record.
static uint32_t attr_words(const IR_Attr *a) {
    switch (a->kind) {
        case AK_ARRAY: case AK_DENSE_I32: return a->n;
        case AK_DICT: case AK_DENSE_I64: return 2 * a->n;
        default: return 0;
    }
}

// Hashes all `n` bytes, four at a time: once per created string, and a
// sample would make same-length strings that share it (data segments)
// collide.
static uint32_t attr_hash_bytes(uint32_t h, const uint8_t *p, uint32_t n) {
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32_t w = (uint32_t)p[i] | ((uint32_t)p[i + 1] << 8) |
                     ((uint32_t)p[i + 2] << 16) | ((uint32_t)p[i + 3] << 24);
        ATTR_MIX(h, w);
    }
    for (; i < n; i++) ATTR_MIX(h, p[i]);
    return h;
}

static uint32_t attr_hash(const IR_Attr *a, const uint32_t *kids) {
    uint32_t h = 0x811C9DC5u ^ ((uint32_t)a->kind << 24);
    ATTR_MIX(h, a->type);
    ATTR_MIX(h, a->n);
    if (a->kind == AK_STRING) return attr_hash_bytes(h, (const uint8_t *)(uintptr_t)a->bits, a->n);
    ATTR_MIX(h, (uint32_t)a->bits);
    ATTR_MIX(h, (uint32_t)(a->bits >> 32));
    uint32_t nw = attr_words(a);
    for (uint32_t i = 0; i < nw; i++) ATTR_MIX(h, kids[i]);
    return h;
}

// The rest of the comparison of two attributes whose kind, type and length
// are equal (see attr_intern).
static bool attr_same_payload(const IR_Attr *a, const uint32_t *ak, const IR_Attr *b, const uint32_t *bk) {
    if (a->kind == AK_STRING) {
        const uint8_t *p = (const uint8_t *)(uintptr_t)a->bits;
        const uint8_t *q = (const uint8_t *)(uintptr_t)b->bits;
        for (uint32_t i = 0; i < a->n; i++)
            if (p[i] != q[i]) return false;
        return true;
    }
    if (a->bits != b->bits) return false;
    uint32_t nw = attr_words(a);
    for (uint32_t i = 0; i < nw; i++)
        if (ak[i] != bk[i]) return false;
    return true;
}

static void amap_alloc(MLIR_Context *c, uint32_t cap) {
    size_t actual = 0;
    c->amap = (uint32_t *)buddy_alloc((size_t)cap * sizeof(uint32_t), &actual);
    if (!c->amap) {
        fprintf(stderr, "MLIR: out of memory growing the attribute table\n");
        abort();
    }
    memset(c->amap, 0, (size_t)cap * sizeof(uint32_t));
    c->amap_cap = cap;
    c->amap_n = 0;
}

static void amap_insert(MLIR_Context *c, uint32_t a, uint32_t h) {
    uint32_t mask = c->amap_cap - 1;
    uint32_t i = h & mask;
    while (c->amap[i] != 0) i = (i + 1) & mask;
    c->amap[i] = a;
    c->amap_n++;
}

static void amap_grow(MLIR_Context *c) {
    uint32_t *old = c->amap;
    amap_alloc(c, c->amap_cap * 2);
    for (uint32_t a = 1; a < c->attrs.n; a++) {
        IR_Attr *r = &ATTRS(c)[a];
        if (r->kind == AK_DISTINCT) continue;
        amap_insert(c, a, attr_hash(r, AX(c) + r->first));
    }
    buddy_free(old);
}

// Interns the candidate `*a`, whose child words (if any) are the last words
// of `ax` from `a->first` on (popped again on a hit) and whose string bytes
// (AK_STRING) are the caller's (copied to the persistent arena on a miss).
static uint32_t attr_intern(MLIR_Context *c, IR_Attr *a) {
    uint32_t h = attr_hash(a, AX(c) + a->first);
    uint32_t mask = c->amap_cap - 1;
    for (uint32_t i = h & mask; c->amap[i] != 0; i = (i + 1) & mask) {
        uint32_t e = c->amap[i];
        const IR_Attr *r = &ATTRS(c)[e];
        if (r->kind == a->kind && r->type == a->type && r->n == a->n &&
            attr_same_payload(r, AX(c) + r->first, a, AX(c) + a->first)) {
            if (attr_words(a)) c->ax.n = a->first;
            return e;
        }
    }
    if (a->kind == AK_STRING) {
        // Own the bytes (an empty string: point at a constant, not at the
        // caller's buffer).
        const char *p = "";
        if (a->n > 0) {
            char *q = arena_new_array(c->persistent, char, a->n);
            memcpy(q, (const void *)(uintptr_t)a->bits, a->n);
            p = q;
        }
        a->bits = (uint64_t)(uintptr_t)p;
    }
    // Grow first: the rehash inserts every attribute already in `attrs`.
    if ((c->amap_n + 1) * 2 > c->amap_cap) amap_grow(c);
    uint32_t idx = vec_push(&c->attrs, 1);
    ATTRS(c)[idx] = *a;
    amap_insert(c, idx, h);
    return idx;
}

// The fast path of attr_intern for a scalar (integer, float, bool, type):
// the same hash and comparison, without building a candidate record.
static uint32_t attr_intern_scalar(MLIR_Context *c, uint32_t kind, uint32_t type, uint64_t bits) {
    uint32_t h = 0x811C9DC5u ^ (kind << 24);
    ATTR_MIX(h, type);
    ATTR_MIX(h, 0);
    ATTR_MIX(h, (uint32_t)bits);
    ATTR_MIX(h, (uint32_t)(bits >> 32));
    uint32_t mask = c->amap_cap - 1;
    for (uint32_t i = h & mask; c->amap[i] != 0; i = (i + 1) & mask) {
        const IR_Attr *r = &ATTRS(c)[c->amap[i]];
        if (r->bits == bits && r->kind == kind && r->type == type && r->n == 0) return c->amap[i];
    }
    if ((c->amap_n + 1) * 2 > c->amap_cap) amap_grow(c);
    uint32_t idx = vec_push(&c->attrs, 1);
    IR_Attr *r = &ATTRS(c)[idx];
    r->kind = (uint8_t)kind;
    r->pad8 = 0;
    r->pad16 = 0;
    r->type = type;
    r->first = 0;
    r->n = 0;
    r->bits = bits;
    r->text = 0;
    r->pad = 0;
    amap_insert(c, idx, h);
    return idx;
}

// An empty candidate of kind `k`.
static void attr_init(IR_Attr *a, AttrKindInternal k) {
    a->kind = (uint8_t)k;
    a->pad8 = 0;
    a->pad16 = 0;
    a->type = 0;
    a->first = 0;
    a->n = 0;
    a->bits = 0;
    a->text = 0;
    a->pad = 0;
}

// The record of attribute handle `h` (element 0, all zero, for the invalid
// handle).
#define ATTR_REC(ctx, h, fn) (&ATTRS(ctx)[ATTR_IDX((ctx), (h), (fn))])

// Context
//
// The context lives in its own persistent arena, together with the names of
// op types and identified structs, so destroying it is one arena_destroy plus
// the IR and type vectors. 64 KiB minus the chunk header keeps the first chunk within one
// 64 KiB buddy block.
MLIR_Context *MLIR_CreateContext(const MLIR_ContextOptions *options) {
    Arena *persistent = arena_create(64 * 1024 - 256);
    if (!persistent) return NULL;
    MLIR_Context *ctx = arena_new(persistent, MLIR_Context);
    *ctx = (MLIR_Context){0};
    ctx->persistent = persistent;
    op_type_table_init(&ctx->op_types, persistent);
    attr_key_table_init(&ctx->keys, persistent);
#ifdef MLIR_API_CHECKS
    ctx->salt = (uint32_t)((((uintptr_t)ctx >> 12) ^ ((uintptr_t)ctx >> 20)) & 0xff) << 20;
#endif
    vec_init(&ctx->ops, (uint32_t)sizeof(IR_Op));
    vec_init(&ctx->vals, (uint32_t)sizeof(IR_Value));
    vec_init(&ctx->opnds, (uint32_t)sizeof(IR_Operand));
    vec_init(&ctx->extra, (uint32_t)sizeof(uint32_t));
    vec_init(&ctx->blocks, (uint32_t)sizeof(IR_Block));
    vec_init(&ctx->regions, (uint32_t)sizeof(IR_Region));
    vec_init(&ctx->op_extra, (uint32_t)sizeof(IR_OpExtra));
    vec_init(&ctx->op_aux, (uint32_t)sizeof(IR_OpAux));
    vec_init(&ctx->names, (uint32_t)sizeof(string));
    vec_init(&ctx->types, (uint32_t)sizeof(IR_Type));
    vec_init(&ctx->tyx, (uint32_t)sizeof(uint32_t));
    vec_init(&ctx->tnames, (uint32_t)sizeof(string));
    tmap_alloc(ctx, 256);
    vec_init(&ctx->attrs, (uint32_t)sizeof(IR_Attr));
    vec_init(&ctx->ax, (uint32_t)sizeof(uint32_t));
    vec_init(&ctx->atext, (uint32_t)sizeof(string));
    amap_alloc(ctx, 1024);
    vec_init(&ctx->locs, (uint32_t)sizeof(IR_Loc));
    vec_init(&ctx->preds, (uint32_t)sizeof(IR_PredEdge));
    attr_key_table_init_strings(&ctx->loc_strings, persistent);
    // Element 1: the shared unknown location.
    {
        uint32_t u = vec_push(&ctx->locs, 1);
        memset(&LOCS(ctx)[u], 0, sizeof(IR_Loc));
        LOCS(ctx)[u].kind = MLIR_LOC_UNKNOWN;
    }
    if (options) {
        ctx->arena = options->arena;
        ctx->no_def_use_tracking = options->no_def_use_tracking;
        if (options->expected_ops) MLIR_SetContextExpectedOps(ctx, options->expected_ops);
    }
    return ctx;
}

// The most MLIR_SetContextExpectedOps reserves, in buddy blocks. The IR
// vectors come from corec's buddy heap, which is a 4 GiB reservation on
// Linux and Windows hosts (and wasm32 linear memory is capped at 4 GiB too):
// a reserved block uses up that heap even where its pages are never touched,
// so an estimate that is too high could exhaust it. A quarter of the heap
// holds a self-host lift's ops, values, operands and attribute words;
// beyond it the vectors double as they fill.
#define IR_RESERVE_BUDGET ((size_t)1 << 30)

// The buddy block a request for `bytes` takes (a power of two, with room for
// the allocator's header).
static size_t buddy_block_bytes(size_t bytes) {
    size_t b = 4096;
    while (b < bytes + 64) b <<= 1;
    return b;
}

// Reserves the vectors that grow with the op count for `n` more ops: per
// op, tinyc's C pipeline makes about 0.8 values, 1.1 operands and 0.75
// attributes (two `extra` words each), and the wasm pipelines 0.4-0.5,
// 0.5-0.7 and 1.4-1.7. The region/successor words (0.05-0.2 per op, also in
// `extra`), blocks, regions and the printer-only vectors are small and just
// double. The estimate shrinks by
// a quarter at a time until the blocks fit IR_RESERVE_BUDGET.
void MLIR_SetContextExpectedOps(MLIR_Context *ctx, size_t n) {
    API_CHECK(ctx != NULL, "MLIR_SetContextExpectedOps", "context is NULL");
    while (n > 0) {
        size_t total = buddy_block_bytes((ctx->ops.n + n) * sizeof(IR_Op)) +
                       buddy_block_bytes((ctx->vals.n + n) * sizeof(IR_Value)) +
                       buddy_block_bytes((ctx->opnds.n + 2 * n) * sizeof(IR_Operand)) +
                       buddy_block_bytes((ctx->extra.n + 3 * n) * sizeof(uint32_t));
        if (total <= IR_RESERVE_BUDGET) break;
        n -= n / 4 ? n / 4 : n;
    }
    if (n == 0) return;
    vec_reserve(&ctx->ops, ctx->ops.n + n);
    vec_reserve(&ctx->vals, ctx->vals.n + n);
    vec_reserve(&ctx->opnds, ctx->opnds.n + 2 * n);
    // Two words per attribute, (key, attribute index).
    vec_reserve(&ctx->extra, ctx->extra.n + 3 * n);
}

void MLIR_DestroyContext(MLIR_Context *ctx) {
    if (!ctx) return;
    vec_free(&ctx->ops);
    vec_free(&ctx->vals);
    vec_free(&ctx->opnds);
    vec_free(&ctx->extra);
    vec_free(&ctx->blocks);
    vec_free(&ctx->regions);
    vec_free(&ctx->op_extra);
    vec_free(&ctx->op_aux);
    vec_free(&ctx->names);
    u32map_free(&ctx->op_extra_map);
    u32map_free(&ctx->op_aux_map);
    u32map_free(&ctx->vloc_map);
    u32map_free(&ctx->vname_map);
    vec_free(&ctx->types);
    vec_free(&ctx->tyx);
    vec_free(&ctx->tnames);
    buddy_free(ctx->tmap);
    vec_free(&ctx->attrs);
    vec_free(&ctx->ax);
    vec_free(&ctx->atext);
    buddy_free(ctx->amap);
    vec_free(&ctx->locs);
    vec_free(&ctx->preds);
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

// `user` is the context.
static void default_diagnostic_handler(void *user, MLIR_DiagSeverity severity,
                                       MLIR_LocationHandle loc, string message) {
    MLIR_Context *ctx = (MLIR_Context *)user;
    const char *sev = "error";
    if (severity == MLIR_DIAG_WARNING) sev = "warning";
    else if (severity == MLIR_DIAG_NOTE) sev = "note";
    else if (severity == MLIR_DIAG_REMARK) sev = "remark";
    if (loc != MLIR_INVALID_HANDLE && MLIR_GetLocationKind(ctx, loc) == MLIR_LOC_FILE) {
        string file = MLIR_GetLocationFileFilename(ctx, loc);
        fprintf(stderr, "%.*s:%d:%d: %s: %.*s\n", (int)file.size, file.str,
                MLIR_GetLocationFileLine(ctx, loc), MLIR_GetLocationFileColumn(ctx, loc),
                sev, (int)message.size, message.str);
    } else if (loc != MLIR_INVALID_HANDLE && MLIR_GetLocationKind(ctx, loc) != MLIR_LOC_UNKNOWN &&
               MLIR_GetLocationOriginalText(ctx, loc).size > 0) {
        // Other locations print in their `loc(...)` form, as upstream's do.
        string text = MLIR_GetLocationOriginalText(ctx, loc);
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
        default_diagnostic_handler(ctx, severity, loc, message);
    if (buf != local) free(buf);
}

static MLIR_NAT_NOINLINE void create_op_extra(MLIR_Context *ctx, uint32_t op,
                                              const MLIR_OpSpec *spec) {
    uint32_t n_succ_ops = 0;
    if (spec->successor_operand_counts) {
        for (uint32_t s = 0; s < spec->n_successors; s++)
            n_succ_ops += spec->successor_operand_counts[s];
    }
    API_CHECK(n_succ_ops <= spec->n_operands, "MLIR_CreateOpFromSpec",
              "successor operand counts exceed n_operands");
    size_t n_extra = (size_t)spec->n_regions + 3 * (size_t)spec->n_successors;
    uint32_t first_extra = (n_extra + 2 * (size_t)spec->n_attrs)
        ? vec_push(&ctx->extra, (uint32_t)(n_extra + 2 * (size_t)spec->n_attrs)) : 0;
    uint32_t first_attr = first_extra + (uint32_t)n_extra;
    uint32_t fresh_regions = (!spec->regions && spec->n_regions > 0)
        ? vec_push(&ctx->regions, spec->n_regions) : 0;
    uint32_t first_pred = spec->n_successors ? vec_push(&ctx->preds, spec->n_successors) : 0;

    IR_Op *o = &OPS(ctx)[op];
    o->n_operands = (uint16_t)(spec->n_operands - n_succ_ops);
    o->first_attr = first_attr;
    o->n_attributes = (uint8_t)spec->n_attrs;
    if (spec->n_regions > 0 || spec->n_successors > 0) {
        uint32_t ei = vec_push(&ctx->op_extra, 1);
        IR_OpExtra *ox = &OPEXTRA(ctx)[ei];
        ox->first_extra = first_extra;
        ox->n_regions = (uint16_t)spec->n_regions;
        ox->n_successors = (uint16_t)spec->n_successors;
        ox->n_succ_operands = (uint16_t)n_succ_ops;
        ox->pad = 0;
        if (spec->n_results == 0) {
            OPS(ctx)[op].first_result = ei;
            OPS(ctx)[op].flags |= (uint8_t)(OPF_HAS_EXTRA | OPF_INLINE_EXTRA);
        } else {
            OPS(ctx)[op].flags |= OPF_HAS_EXTRA;
            u32map_put(&ctx->op_extra_map, op, ei);
        }
    }

    uint32_t *ex = EXTRA(ctx) + first_extra;
    if (spec->regions) {
        for (uint32_t i = 0; i < spec->n_regions; i++) {
            uint32_t r = REGION_IDX(ctx, spec->regions[i], "MLIR_CreateOpFromSpec");
            ex[i] = r;
            if (r) REGIONS(ctx)[r].parent_op = op;
        }
    } else if (spec->n_regions > 0) {
        for (uint32_t i = 0; i < spec->n_regions; i++) {
            uint32_t r = fresh_regions + i;
            IR_Region *reg = &REGIONS(ctx)[r];
            reg->blocks = NULL;
            reg->n_blocks = 0;
            reg->cap_blocks = 0;
            reg->parent_op = op;
            ex[i] = r;
        }
    }
    ex += spec->n_regions;
    for (uint32_t s = 0; s < spec->n_successors; s++) {
        uint32_t cnt = spec->successor_operand_counts ? spec->successor_operand_counts[s] : 0;
        uint32_t tb = BLOCK_IDX(ctx, spec->successors[s], "MLIR_CreateOpFromSpec");
        uint32_t pe_idx = first_pred + s;
        ex[3 * s] = tb;
        ex[3 * s + 1] = cnt;
        ex[3 * s + 2] = pe_idx;
        IR_PredEdge *pe = &PREDS(ctx)[pe_idx];
        pe->owner = op;
        pe->succ_idx = s;
        pe->target_block = tb;
        pe->next = pe->prev = 0;
        if (tb) pred_push(ctx, pe_idx);
    }

    uint32_t *pa = EXTRA(ctx) + first_attr;
    for (uint32_t i = 0; i < spec->n_attrs; i++) {
        API_CHECK(spec->attrs[i].key != MLIR_KEY_INVALID && spec->attrs[i].key < ctx->keys.n_keys &&
                  spec->attrs[i].value != MLIR_INVALID_HANDLE,
                  "MLIR_CreateOpFromSpec", "bad attribute key or value");
        pa[2 * i] = spec->attrs[i].key;
        pa[2 * i + 1] = ATTR_IDX(ctx, spec->attrs[i].value, "MLIR_CreateOpFromSpec");
    }
}

static inline uint32_t create_op_from_spec_impl(MLIR_Context *ctx, const MLIR_OpSpec *spec,
                                                MLIR_LocationHandle loc_override) {
    MLIR_OpType type = spec->type;
    API_CHECK(type != OP_TYPE_UNREGISTERED && (uint32_t)type < ctx->op_types.n_types &&
                  (uint32_t)type < 0xffffu,
              "MLIR_CreateOpFromSpec", "not an op type of this context");
    API_CHECK(spec->n_operands <= 0xffff && spec->n_results <= 0xffff &&
                  spec->n_attrs <= 0xff && spec->n_regions <= 0xffff && spec->n_successors <= 0xffff,
              "MLIR_CreateOpFromSpec", "too many operands, results, attributes, regions or successors");

    uint32_t op = vec_push(&ctx->ops, 1);
    uint32_t first_result = spec->n_results ? vec_push(&ctx->vals, spec->n_results) : 0;
    uint32_t first_operand = spec->n_operands ? vec_push(&ctx->opnds, spec->n_operands) : 0;

    bool track = !ctx->no_def_use_tracking;
    MLIR_LocationHandle loc_h = (spec->loc != MLIR_INVALID_HANDLE) ? spec->loc : loc_override;
    uint32_t loc_idx = LOC_IDX(ctx, loc_h, "MLIR_CreateOpFromSpec");

    IR_Op *o = &OPS(ctx)[op];
    o->op_type = (uint16_t)type;
    o->flags = track ? OPF_TRACKED : 0;
    o->n_attributes = 0;
    o->n_operands = (uint16_t)spec->n_operands;
    o->n_results = (uint16_t)spec->n_results;
    o->parent_block = 0;
    o->block_pos = 0;
    o->first_operand = first_operand;
    o->first_result = first_result;
    o->first_attr = 0;
    o->location = loc_idx;

    if (spec->n_results > 0) {
        IR_Value *res = VALS(ctx) + first_result;
        uint32_t track_bit = track ? VAL_TRACKED : 0;
        for (uint32_t i = 0; i < spec->n_results; i++) {
            res[i].def = op;
            res[i].index = (i & VAL_INDEX_MASK) | track_bit;
            res[i].type = spec->result_types
                ? TYPE_IDX(ctx, spec->result_types[i], "MLIR_CreateOpFromSpec") : 0;
            res[i].n_uses = 0;
            res[i].first_use = 0;
        }
    }

    if (spec->n_operands > 0) {
        IR_Operand *rec = OPNDS(ctx);
        IR_Value *vals = VALS(ctx);
        for (uint32_t i = 0; i < spec->n_operands; i++) {
            uint32_t u = first_operand + i;
            uint32_t v = VAL_IDX(ctx, spec->operands[i], "MLIR_CreateOpFromSpec");
            rec[u].value = v;
            rec[u].owner = opnd_pack_owner(op, i);
            rec[u].next = 0;
            rec[u].prev = 0;
            if (MLIR_NAT_LIKELY(track && v != 0 && (vals[v].index & VAL_TRACKED))) {
                uint32_t head = vals[v].first_use;
                rec[u].next = head;
                if (head) rec[head].prev = u;
                vals[v].first_use = u;
                vals[v].n_uses++;
            }
        }
    }

    if (MLIR_NAT_UNLIKELY((spec->n_regions | spec->n_successors) != 0)) {
        create_op_extra(ctx, op, spec);
    } else if (spec->n_attrs > 0) {
        uint32_t first_attr = vec_push(&ctx->extra, 2u * (uint32_t)spec->n_attrs);
        o->first_attr = first_attr;
        o->n_attributes = (uint8_t)spec->n_attrs;
        uint32_t *pa = EXTRA(ctx) + first_attr;
        for (uint32_t i = 0; i < spec->n_attrs; i++) {
            API_CHECK(spec->attrs[i].key != MLIR_KEY_INVALID && spec->attrs[i].key < ctx->keys.n_keys &&
                      spec->attrs[i].value != MLIR_INVALID_HANDLE,
                      "MLIR_CreateOpFromSpec", "bad attribute key or value");
            pa[2 * i] = spec->attrs[i].key;
            pa[2 * i + 1] = ATTR_IDX(ctx, spec->attrs[i].value, "MLIR_CreateOpFromSpec");
        }
    }

    return op;
}

MLIR_OpHandle MLIR_CreateOpFromSpec(MLIR_Context *ctx, const MLIR_OpSpec *spec) {
    API_CHECK(ctx != NULL, "MLIR_CreateOpFromSpec", "context is NULL");
    API_CHECK(spec != NULL, "MLIR_CreateOpFromSpec", "spec is NULL");
    return OP_H(ctx, create_op_from_spec_impl(ctx, spec, MLIR_INVALID_HANDLE));
}

static void block_insert_op(MLIR_Context *ctx, uint32_t b, uint32_t o, size_t idx);

void MLIR_InsertOpBefore(MLIR_Context *ctx, MLIR_BlockHandle bh,
                         MLIR_OpHandle oh, MLIR_OpHandle before_h) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_InsertOpBefore");
    uint32_t o = OP_IDX(ctx, oh, "MLIR_InsertOpBefore");
    API_CHECK(b != 0, "MLIR_InsertOpBefore", "block handle is NULL");
    CHECK_OP(ctx, o, "MLIR_InsertOpBefore");
    if (OPS(ctx)[o].parent_block != 0) MLIR_DetachOp(ctx, oh);
    IR_Block *blk = &BLOCKS(ctx)[b];
    size_t idx = blk->n_ops;
    if (before_h != MLIR_INVALID_HANDLE) {
        uint32_t bef = OP_IDX(ctx, before_h, "MLIR_InsertOpBefore");
        CHECK_OP(ctx, bef, "MLIR_InsertOpBefore");
        API_CHECK(OPS(ctx)[bef].parent_block == b, "MLIR_InsertOpBefore",
                  "before op is not in target block");
        idx = OPS(ctx)[bef].block_pos;
    }
    block_insert_op(ctx, b, o, idx);
}

MLIR_Builder MLIR_BuilderAfter(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_BuilderAfter");
    CHECK_OP(ctx, o, "MLIR_BuilderAfter");
    uint32_t b = OPS(ctx)[o].parent_block;
    API_CHECK(b != 0, "MLIR_BuilderAfter", "op has no parent block");
    IR_Block *blk = &BLOCKS(ctx)[b];
    uint32_t pos = OPS(ctx)[o].block_pos;
    MLIR_Builder res;
    res.ctx = ctx;
    res.block = BLOCK_H(ctx, b);
    res.before = (pos + 1 < blk->n_ops) ? OP_H(ctx, blk->ops[pos + 1]) : MLIR_INVALID_HANDLE;
    res.loc = MLIR_INVALID_HANDLE;
    return res;
}

MLIR_OpHandle MLIR_BuildOp(MLIR_Builder *b, const MLIR_OpSpec *spec) {
    API_CHECK(b != NULL && b->ctx != NULL && spec != NULL, "MLIR_BuildOp", "NULL argument");
    MLIR_Context *ctx = b->ctx;
    uint32_t op = create_op_from_spec_impl(ctx, spec, b->loc);
    if (MLIR_NAT_LIKELY(b->block != MLIR_INVALID_HANDLE)) {
        if (MLIR_NAT_LIKELY(b->before == MLIR_INVALID_HANDLE)) {
            uint32_t bi = BLOCK_IDX(ctx, b->block, "MLIR_BuildOp");
            API_CHECK(bi != 0, "MLIR_BuildOp", "block handle is NULL");
            IR_Block *blk = &BLOCKS(ctx)[bi];
            uint32_t pos = blk->n_ops;
            if (MLIR_NAT_UNLIKELY(pos >= blk->cap_ops))
                blk->ops = u32_reserve(ctx->arena, blk->ops, pos, &blk->cap_ops, (size_t)pos + 1);
            blk->ops[pos] = op;
            blk->n_ops = pos + 1;
            IR_Op *o = &OPS(ctx)[op];
            o->parent_block = bi;
            o->block_pos = pos;
        } else {
            MLIR_InsertOpBefore(ctx, b->block, OP_H(ctx, op), b->before);
        }
    }
    return OP_H(ctx, op);
}

static inline void link_operand_tracked(IR_Operand *rec, IR_Value *vals, uint32_t u,
                                        uint32_t v, uint32_t packed_owner) {
    rec[u].value = v;
    rec[u].owner = packed_owner;
    rec[u].prev = 0;
    if (MLIR_NAT_LIKELY(v != 0 && (vals[v].index & VAL_TRACKED))) {
        uint32_t head = vals[v].first_use;
        rec[u].next = head;
        if (head) rec[head].prev = u;
        vals[v].first_use = u;
        vals[v].n_uses++;
    } else {
        rec[u].next = 0;
    }
}

static MLIR_NAT_NOINLINE MLIR_ValueHandle build_binary_slow(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x, MLIR_ValueHandle y) {
    MLIR_ValueHandle ops[2];
    ops[0] = x;
    ops[1] = y;
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = t;
    s.operands = ops;
    s.n_operands = 2;
    s.result_types = &rt;
    s.n_results = 1;
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

static MLIR_NAT_NOINLINE MLIR_ValueHandle build_unary_slow(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x) {
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = t;
    s.operands = &x;
    s.n_operands = 1;
    s.result_types = &rt;
    s.n_results = 1;
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

MLIR_ValueHandle MLIR_BuildBinary(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x, MLIR_ValueHandle y) {
    API_CHECK(b != NULL && b->ctx != NULL, "MLIR_BuildBinary", "NULL argument");
    MLIR_Context *ctx = b->ctx;
    API_CHECK(t != OP_TYPE_UNREGISTERED && (uint32_t)t < ctx->op_types.n_types && (uint32_t)t < 0xffffu,
              "MLIR_BuildBinary", "not an op type of this context");

    uint32_t op = ctx->ops.n;
    uint32_t res = ctx->vals.n;
    uint32_t u0 = ctx->opnds.n;
    uint32_t bi = BLOCK_IDX(ctx, b->block, "MLIR_BuildBinary");
    if (MLIR_NAT_UNLIKELY(ctx->no_def_use_tracking ||
                          op >= ctx->ops.cap || res >= ctx->vals.cap || u0 + 2 > ctx->opnds.cap ||
                          bi == 0 || b->before != MLIR_INVALID_HANDLE)) {
        return build_binary_slow(b, t, rt, x, y);
    }
    IR_Block *blk = &BLOCKS(ctx)[bi];
    uint32_t pos = blk->n_ops;
    if (MLIR_NAT_UNLIKELY(pos >= blk->cap_ops)) {
        return build_binary_slow(b, t, rt, x, y);
    }

    ctx->ops.n = op + 1;
    ctx->vals.n = res + 1;
    ctx->opnds.n = u0 + 2;
    blk->ops[pos] = op;
    blk->n_ops = pos + 1;

    IR_Op *o = &OPS(ctx)[op];
    o->op_type = (uint16_t)t;
    o->flags = OPF_TRACKED;
    o->n_attributes = 0;
    o->n_operands = 2;
    o->n_results = 1;
    o->parent_block = bi;
    o->block_pos = pos;
    o->first_operand = u0;
    o->first_result = res;
    o->first_attr = 0;
    o->location = LOC_IDX(ctx, b->loc, "MLIR_BuildBinary");

    IR_Value *vals = VALS(ctx);
    IR_Value *rv = &vals[res];
    rv->type = TYPE_IDX(ctx, rt, "MLIR_BuildBinary");
    rv->def = op;
    rv->index = VAL_TRACKED;
    rv->first_use = 0;
    rv->n_uses = 0;

    IR_Operand *rec = OPNDS(ctx);
    link_operand_tracked(rec, vals, u0, VAL_IDX(ctx, x, "MLIR_BuildBinary"), op);
    link_operand_tracked(rec, vals, u0 + 1, VAL_IDX(ctx, y, "MLIR_BuildBinary"), op | (1u << OPND_IDX_SHIFT));
    return VAL_H(ctx, res);
}

MLIR_ValueHandle MLIR_BuildUnary(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x) {
    API_CHECK(b != NULL && b->ctx != NULL, "MLIR_BuildUnary", "NULL argument");
    MLIR_Context *ctx = b->ctx;
    API_CHECK(t != OP_TYPE_UNREGISTERED && (uint32_t)t < ctx->op_types.n_types && (uint32_t)t < 0xffffu,
              "MLIR_BuildUnary", "not an op type of this context");

    uint32_t op = ctx->ops.n;
    uint32_t res = ctx->vals.n;
    uint32_t u0 = ctx->opnds.n;
    uint32_t bi = BLOCK_IDX(ctx, b->block, "MLIR_BuildUnary");
    if (MLIR_NAT_UNLIKELY(ctx->no_def_use_tracking ||
                          op >= ctx->ops.cap || res >= ctx->vals.cap || u0 + 1 > ctx->opnds.cap ||
                          bi == 0 || b->before != MLIR_INVALID_HANDLE)) {
        return build_unary_slow(b, t, rt, x);
    }
    IR_Block *blk = &BLOCKS(ctx)[bi];
    uint32_t pos = blk->n_ops;
    if (MLIR_NAT_UNLIKELY(pos >= blk->cap_ops)) {
        return build_unary_slow(b, t, rt, x);
    }

    ctx->ops.n = op + 1;
    ctx->vals.n = res + 1;
    ctx->opnds.n = u0 + 1;
    blk->ops[pos] = op;
    blk->n_ops = pos + 1;

    IR_Op *o = &OPS(ctx)[op];
    o->op_type = (uint16_t)t;
    o->flags = OPF_TRACKED;
    o->n_attributes = 0;
    o->n_operands = 1;
    o->n_results = 1;
    o->parent_block = bi;
    o->block_pos = pos;
    o->first_operand = u0;
    o->first_result = res;
    o->first_attr = 0;
    o->location = LOC_IDX(ctx, b->loc, "MLIR_BuildUnary");

    IR_Value *vals = VALS(ctx);
    IR_Value *rv = &vals[res];
    rv->type = TYPE_IDX(ctx, rt, "MLIR_BuildUnary");
    rv->def = op;
    rv->index = VAL_TRACKED;
    rv->first_use = 0;
    rv->n_uses = 0;

    IR_Operand *rec = OPNDS(ctx);
    link_operand_tracked(rec, vals, u0, VAL_IDX(ctx, x, "MLIR_BuildUnary"), op);
    return VAL_H(ctx, res);
}

static MLIR_NAT_NOINLINE MLIR_OpHandle build_void_binary_slow(MLIR_Builder *b, MLIR_OpType t,
        MLIR_ValueHandle x, MLIR_ValueHandle y) {
    MLIR_ValueHandle ops[2] = { x, y };
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = t;
    s.operands = ops;
    s.n_operands = 2;
    return MLIR_BuildOp(b, &s);
}

MLIR_OpHandle MLIR_BuildVoidBinary(MLIR_Builder *b, MLIR_OpType t,
        MLIR_ValueHandle x, MLIR_ValueHandle y) {
    API_CHECK(b != NULL && b->ctx != NULL, "MLIR_BuildVoidBinary", "NULL argument");
    MLIR_Context *ctx = b->ctx;
    API_CHECK(t != OP_TYPE_UNREGISTERED && (uint32_t)t < ctx->op_types.n_types && (uint32_t)t < 0xffffu,
              "MLIR_BuildVoidBinary", "not an op type of this context");

    uint32_t op = ctx->ops.n;
    uint32_t u0 = ctx->opnds.n;
    uint32_t bi = BLOCK_IDX(ctx, b->block, "MLIR_BuildVoidBinary");
    if (MLIR_NAT_UNLIKELY(ctx->no_def_use_tracking ||
                          op >= ctx->ops.cap || u0 + 2 > ctx->opnds.cap ||
                          bi == 0 || b->before != MLIR_INVALID_HANDLE)) {
        return build_void_binary_slow(b, t, x, y);
    }
    IR_Block *blk = &BLOCKS(ctx)[bi];
    uint32_t pos = blk->n_ops;
    if (MLIR_NAT_UNLIKELY(pos >= blk->cap_ops)) {
        return build_void_binary_slow(b, t, x, y);
    }

    ctx->ops.n = op + 1;
    ctx->opnds.n = u0 + 2;
    blk->ops[pos] = op;
    blk->n_ops = pos + 1;

    IR_Op *o = &OPS(ctx)[op];
    o->op_type = (uint16_t)t;
    o->flags = OPF_TRACKED;
    o->n_attributes = 0;
    o->n_operands = 2;
    o->n_results = 0;
    o->parent_block = bi;
    o->block_pos = pos;
    o->first_operand = u0;
    o->first_result = 0;
    o->first_attr = 0;
    o->location = LOC_IDX(ctx, b->loc, "MLIR_BuildVoidBinary");

    IR_Value *vals = VALS(ctx);
    IR_Operand *rec = OPNDS(ctx);
    link_operand_tracked(rec, vals, u0, VAL_IDX(ctx, x, "MLIR_BuildVoidBinary"), op);
    link_operand_tracked(rec, vals, u0 + 1, VAL_IDX(ctx, y, "MLIR_BuildVoidBinary"), op | (1u << OPND_IDX_SHIFT));
    return OP_H(ctx, op);
}

static MLIR_NAT_NOINLINE MLIR_ValueHandle build_nullary_attr_slow(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_AttrKey key, MLIR_AttributeHandle val) {
    MLIR_NamedAttr a = MLIR_Named(key, val);
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = t;
    s.result_types = &rt;
    s.n_results = 1;
    s.attrs = &a;
    s.n_attrs = 1;
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

MLIR_ValueHandle MLIR_BuildNullaryAttr(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_AttrKey key, MLIR_AttributeHandle val) {
    API_CHECK(b != NULL && b->ctx != NULL, "MLIR_BuildNullaryAttr", "NULL argument");
    MLIR_Context *ctx = b->ctx;
    API_CHECK(t != OP_TYPE_UNREGISTERED && (uint32_t)t < ctx->op_types.n_types && (uint32_t)t < 0xffffu,
              "MLIR_BuildNullaryAttr", "not an op type of this context");
    API_CHECK(key != MLIR_KEY_INVALID && key < ctx->keys.n_keys && val != MLIR_INVALID_HANDLE,
              "MLIR_BuildNullaryAttr", "bad attribute key or value");

    uint32_t op = ctx->ops.n;
    uint32_t res = ctx->vals.n;
    uint32_t fa = ctx->extra.n;
    uint32_t bi = BLOCK_IDX(ctx, b->block, "MLIR_BuildNullaryAttr");
    if (MLIR_NAT_UNLIKELY(ctx->no_def_use_tracking ||
                          op >= ctx->ops.cap || res >= ctx->vals.cap || fa + 2 > ctx->extra.cap ||
                          bi == 0 || b->before != MLIR_INVALID_HANDLE)) {
        return build_nullary_attr_slow(b, t, rt, key, val);
    }
    IR_Block *blk = &BLOCKS(ctx)[bi];
    uint32_t pos = blk->n_ops;
    if (MLIR_NAT_UNLIKELY(pos >= blk->cap_ops)) {
        return build_nullary_attr_slow(b, t, rt, key, val);
    }

    ctx->ops.n = op + 1;
    ctx->vals.n = res + 1;
    ctx->extra.n = fa + 2;
    blk->ops[pos] = op;
    blk->n_ops = pos + 1;

    IR_Op *o = &OPS(ctx)[op];
    o->op_type = (uint16_t)t;
    o->flags = OPF_TRACKED;
    o->n_attributes = 1;
    o->n_operands = 0;
    o->n_results = 1;
    o->parent_block = bi;
    o->block_pos = pos;
    o->first_operand = 0;
    o->first_result = res;
    o->first_attr = fa;
    o->location = LOC_IDX(ctx, b->loc, "MLIR_BuildNullaryAttr");

    IR_Value *rv = &VALS(ctx)[res];
    rv->type = TYPE_IDX(ctx, rt, "MLIR_BuildNullaryAttr");
    rv->def = op;
    rv->index = VAL_TRACKED;
    rv->first_use = 0;
    rv->n_uses = 0;

    uint32_t *pa = EXTRA(ctx) + fa;
    pa[0] = key;
    pa[1] = ATTR_IDX(ctx, val, "MLIR_BuildNullaryAttr");
    return VAL_H(ctx, res);
}

static MLIR_NAT_NOINLINE MLIR_ValueHandle build_unary_attr_slow(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x,
        MLIR_AttrKey key, MLIR_AttributeHandle val) {
    MLIR_NamedAttr a = MLIR_Named(key, val);
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = t;
    s.operands = &x;
    s.n_operands = 1;
    s.result_types = &rt;
    s.n_results = 1;
    s.attrs = &a;
    s.n_attrs = 1;
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

MLIR_ValueHandle MLIR_BuildUnaryAttr(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x,
        MLIR_AttrKey key, MLIR_AttributeHandle val) {
    API_CHECK(b != NULL && b->ctx != NULL, "MLIR_BuildUnaryAttr", "NULL argument");
    MLIR_Context *ctx = b->ctx;
    API_CHECK(t != OP_TYPE_UNREGISTERED && (uint32_t)t < ctx->op_types.n_types && (uint32_t)t < 0xffffu,
              "MLIR_BuildUnaryAttr", "not an op type of this context");
    API_CHECK(key != MLIR_KEY_INVALID && key < ctx->keys.n_keys && val != MLIR_INVALID_HANDLE,
              "MLIR_BuildUnaryAttr", "bad attribute key or value");

    uint32_t op = ctx->ops.n;
    uint32_t res = ctx->vals.n;
    uint32_t u0 = ctx->opnds.n;
    uint32_t fa = ctx->extra.n;
    uint32_t bi = BLOCK_IDX(ctx, b->block, "MLIR_BuildUnaryAttr");
    if (MLIR_NAT_UNLIKELY(ctx->no_def_use_tracking ||
                          op >= ctx->ops.cap || res >= ctx->vals.cap ||
                          u0 + 1 > ctx->opnds.cap || fa + 2 > ctx->extra.cap ||
                          bi == 0 || b->before != MLIR_INVALID_HANDLE)) {
        return build_unary_attr_slow(b, t, rt, x, key, val);
    }
    IR_Block *blk = &BLOCKS(ctx)[bi];
    uint32_t pos = blk->n_ops;
    if (MLIR_NAT_UNLIKELY(pos >= blk->cap_ops)) {
        return build_unary_attr_slow(b, t, rt, x, key, val);
    }

    ctx->ops.n = op + 1;
    ctx->vals.n = res + 1;
    ctx->opnds.n = u0 + 1;
    ctx->extra.n = fa + 2;
    blk->ops[pos] = op;
    blk->n_ops = pos + 1;

    IR_Op *o = &OPS(ctx)[op];
    o->op_type = (uint16_t)t;
    o->flags = OPF_TRACKED;
    o->n_attributes = 1;
    o->n_operands = 1;
    o->n_results = 1;
    o->parent_block = bi;
    o->block_pos = pos;
    o->first_operand = u0;
    o->first_result = res;
    o->first_attr = fa;
    o->location = LOC_IDX(ctx, b->loc, "MLIR_BuildUnaryAttr");

    IR_Value *vals = VALS(ctx);
    IR_Value *rv = &vals[res];
    rv->type = TYPE_IDX(ctx, rt, "MLIR_BuildUnaryAttr");
    rv->def = op;
    rv->index = VAL_TRACKED;
    rv->first_use = 0;
    rv->n_uses = 0;

    IR_Operand *rec = OPNDS(ctx);
    link_operand_tracked(rec, vals, u0, VAL_IDX(ctx, x, "MLIR_BuildUnaryAttr"), op);

    uint32_t *pa = EXTRA(ctx) + fa;
    pa[0] = key;
    pa[1] = ATTR_IDX(ctx, val, "MLIR_BuildUnaryAttr");
    return VAL_H(ctx, res);
}

static MLIR_NAT_NOINLINE MLIR_ValueHandle build_binary_attr_slow(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x, MLIR_ValueHandle y,
        MLIR_AttrKey key, MLIR_AttributeHandle val) {
    MLIR_ValueHandle ops[2] = { x, y };
    MLIR_NamedAttr a = MLIR_Named(key, val);
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = t;
    s.operands = ops;
    s.n_operands = 2;
    s.result_types = &rt;
    s.n_results = 1;
    s.attrs = &a;
    s.n_attrs = 1;
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

MLIR_ValueHandle MLIR_BuildBinaryAttr(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x, MLIR_ValueHandle y,
        MLIR_AttrKey key, MLIR_AttributeHandle val) {
    API_CHECK(b != NULL && b->ctx != NULL, "MLIR_BuildBinaryAttr", "NULL argument");
    MLIR_Context *ctx = b->ctx;
    API_CHECK(t != OP_TYPE_UNREGISTERED && (uint32_t)t < ctx->op_types.n_types && (uint32_t)t < 0xffffu,
              "MLIR_BuildBinaryAttr", "not an op type of this context");
    API_CHECK(key != MLIR_KEY_INVALID && key < ctx->keys.n_keys && val != MLIR_INVALID_HANDLE,
              "MLIR_BuildBinaryAttr", "bad attribute key or value");

    uint32_t op = ctx->ops.n;
    uint32_t res = ctx->vals.n;
    uint32_t u0 = ctx->opnds.n;
    uint32_t fa = ctx->extra.n;
    uint32_t bi = BLOCK_IDX(ctx, b->block, "MLIR_BuildBinaryAttr");
    if (MLIR_NAT_UNLIKELY(ctx->no_def_use_tracking ||
                          op >= ctx->ops.cap || res >= ctx->vals.cap ||
                          u0 + 2 > ctx->opnds.cap || fa + 2 > ctx->extra.cap ||
                          bi == 0 || b->before != MLIR_INVALID_HANDLE)) {
        return build_binary_attr_slow(b, t, rt, x, y, key, val);
    }
    IR_Block *blk = &BLOCKS(ctx)[bi];
    uint32_t pos = blk->n_ops;
    if (MLIR_NAT_UNLIKELY(pos >= blk->cap_ops)) {
        return build_binary_attr_slow(b, t, rt, x, y, key, val);
    }

    ctx->ops.n = op + 1;
    ctx->vals.n = res + 1;
    ctx->opnds.n = u0 + 2;
    ctx->extra.n = fa + 2;
    blk->ops[pos] = op;
    blk->n_ops = pos + 1;

    IR_Op *o = &OPS(ctx)[op];
    o->op_type = (uint16_t)t;
    o->flags = OPF_TRACKED;
    o->n_attributes = 1;
    o->n_operands = 2;
    o->n_results = 1;
    o->parent_block = bi;
    o->block_pos = pos;
    o->first_operand = u0;
    o->first_result = res;
    o->first_attr = fa;
    o->location = LOC_IDX(ctx, b->loc, "MLIR_BuildBinaryAttr");

    IR_Value *vals = VALS(ctx);
    IR_Value *rv = &vals[res];
    rv->type = TYPE_IDX(ctx, rt, "MLIR_BuildBinaryAttr");
    rv->def = op;
    rv->index = VAL_TRACKED;
    rv->first_use = 0;
    rv->n_uses = 0;

    IR_Operand *rec = OPNDS(ctx);
    link_operand_tracked(rec, vals, u0, VAL_IDX(ctx, x, "MLIR_BuildBinaryAttr"), op);
    link_operand_tracked(rec, vals, u0 + 1, VAL_IDX(ctx, y, "MLIR_BuildBinaryAttr"), op | (1u << OPND_IDX_SHIFT));

    uint32_t *pa = EXTRA(ctx) + fa;
    pa[0] = key;
    pa[1] = ATTR_IDX(ctx, val, "MLIR_BuildBinaryAttr");
    return VAL_H(ctx, res);
}

static MLIR_NAT_NOINLINE MLIR_ValueHandle build_unary2attr_slow(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x,
        MLIR_AttrKey k0, MLIR_AttributeHandle v0,
        MLIR_AttrKey k1, MLIR_AttributeHandle v1) {
    MLIR_NamedAttr a[2];
    a[0] = MLIR_Named(k0, v0);
    a[1] = MLIR_Named(k1, v1);
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = t;
    s.operands = &x;
    s.n_operands = 1;
    s.result_types = &rt;
    s.n_results = 1;
    s.attrs = a;
    s.n_attrs = 2;
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

MLIR_ValueHandle MLIR_BuildUnary2Attr(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x,
        MLIR_AttrKey k0, MLIR_AttributeHandle v0,
        MLIR_AttrKey k1, MLIR_AttributeHandle v1) {
    API_CHECK(b != NULL && b->ctx != NULL, "MLIR_BuildUnary2Attr", "NULL argument");
    MLIR_Context *ctx = b->ctx;
    API_CHECK(t != OP_TYPE_UNREGISTERED && (uint32_t)t < ctx->op_types.n_types && (uint32_t)t < 0xffffu,
              "MLIR_BuildUnary2Attr", "not an op type of this context");

    uint32_t op = ctx->ops.n;
    uint32_t res = ctx->vals.n;
    uint32_t u0 = ctx->opnds.n;
    uint32_t fa = ctx->extra.n;
    uint32_t bi = BLOCK_IDX(ctx, b->block, "MLIR_BuildUnary2Attr");
    if (MLIR_NAT_UNLIKELY(ctx->no_def_use_tracking ||
                          op >= ctx->ops.cap || res >= ctx->vals.cap ||
                          u0 + 1 > ctx->opnds.cap || fa + 4 > ctx->extra.cap ||
                          bi == 0 || b->before != MLIR_INVALID_HANDLE)) {
        return build_unary2attr_slow(b, t, rt, x, k0, v0, k1, v1);
    }
    IR_Block *blk = &BLOCKS(ctx)[bi];
    uint32_t pos = blk->n_ops;
    if (MLIR_NAT_UNLIKELY(pos >= blk->cap_ops)) {
        return build_unary2attr_slow(b, t, rt, x, k0, v0, k1, v1);
    }

    ctx->ops.n = op + 1;
    ctx->vals.n = res + 1;
    ctx->opnds.n = u0 + 1;
    ctx->extra.n = fa + 4;
    blk->ops[pos] = op;
    blk->n_ops = pos + 1;

    IR_Op *o = &OPS(ctx)[op];
    o->op_type = (uint16_t)t;
    o->flags = OPF_TRACKED;
    o->n_attributes = 2;
    o->n_operands = 1;
    o->n_results = 1;
    o->parent_block = bi;
    o->block_pos = pos;
    o->first_operand = u0;
    o->first_result = res;
    o->first_attr = fa;
    o->location = LOC_IDX(ctx, b->loc, "MLIR_BuildUnary2Attr");

    IR_Value *vals = VALS(ctx);
    IR_Value *rv = &vals[res];
    rv->type = TYPE_IDX(ctx, rt, "MLIR_BuildUnary2Attr");
    rv->def = op;
    rv->index = VAL_TRACKED;
    rv->first_use = 0;
    rv->n_uses = 0;

    link_operand_tracked(OPNDS(ctx), vals, u0, VAL_IDX(ctx, x, "MLIR_BuildUnary2Attr"), op);

    uint32_t *pa = EXTRA(ctx) + fa;
    pa[0] = k0; pa[1] = ATTR_IDX(ctx, v0, "MLIR_BuildUnary2Attr");
    pa[2] = k1; pa[3] = ATTR_IDX(ctx, v1, "MLIR_BuildUnary2Attr");
    return VAL_H(ctx, res);
}

static MLIR_NAT_NOINLINE MLIR_ValueHandle build_binary2attr_slow(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x, MLIR_ValueHandle y,
        MLIR_AttrKey k0, MLIR_AttributeHandle v0,
        MLIR_AttrKey k1, MLIR_AttributeHandle v1) {
    MLIR_ValueHandle ops[2] = { x, y };
    MLIR_NamedAttr a[2];
    a[0] = MLIR_Named(k0, v0);
    a[1] = MLIR_Named(k1, v1);
    MLIR_OpSpec s = { OP_TYPE_UNREGISTERED };
    s.type = t;
    s.operands = ops;
    s.n_operands = 2;
    s.result_types = &rt;
    s.n_results = 1;
    s.attrs = a;
    s.n_attrs = 2;
    return MLIR_GetOpResult(b->ctx, MLIR_BuildOp(b, &s), 0);
}

MLIR_ValueHandle MLIR_BuildBinary2Attr(MLIR_Builder *b, MLIR_OpType t,
        MLIR_TypeHandle rt, MLIR_ValueHandle x, MLIR_ValueHandle y,
        MLIR_AttrKey k0, MLIR_AttributeHandle v0,
        MLIR_AttrKey k1, MLIR_AttributeHandle v1) {
    API_CHECK(b != NULL && b->ctx != NULL, "MLIR_BuildBinary2Attr", "NULL argument");
    MLIR_Context *ctx = b->ctx;
    API_CHECK(t != OP_TYPE_UNREGISTERED && (uint32_t)t < ctx->op_types.n_types && (uint32_t)t < 0xffffu,
              "MLIR_BuildBinary2Attr", "not an op type of this context");

    uint32_t op = ctx->ops.n;
    uint32_t res = ctx->vals.n;
    uint32_t u0 = ctx->opnds.n;
    uint32_t fa = ctx->extra.n;
    uint32_t bi = BLOCK_IDX(ctx, b->block, "MLIR_BuildBinary2Attr");
    if (MLIR_NAT_UNLIKELY(ctx->no_def_use_tracking ||
                          op >= ctx->ops.cap || res >= ctx->vals.cap ||
                          u0 + 2 > ctx->opnds.cap || fa + 4 > ctx->extra.cap ||
                          bi == 0 || b->before != MLIR_INVALID_HANDLE)) {
        return build_binary2attr_slow(b, t, rt, x, y, k0, v0, k1, v1);
    }
    IR_Block *blk = &BLOCKS(ctx)[bi];
    uint32_t pos = blk->n_ops;
    if (MLIR_NAT_UNLIKELY(pos >= blk->cap_ops)) {
        return build_binary2attr_slow(b, t, rt, x, y, k0, v0, k1, v1);
    }

    ctx->ops.n = op + 1;
    ctx->vals.n = res + 1;
    ctx->opnds.n = u0 + 2;
    ctx->extra.n = fa + 4;
    blk->ops[pos] = op;
    blk->n_ops = pos + 1;

    IR_Op *o = &OPS(ctx)[op];
    o->op_type = (uint16_t)t;
    o->flags = OPF_TRACKED;
    o->n_attributes = 2;
    o->n_operands = 2;
    o->n_results = 1;
    o->parent_block = bi;
    o->block_pos = pos;
    o->first_operand = u0;
    o->first_result = res;
    o->first_attr = fa;
    o->location = LOC_IDX(ctx, b->loc, "MLIR_BuildBinary2Attr");

    IR_Value *vals = VALS(ctx);
    IR_Value *rv = &vals[res];
    rv->type = TYPE_IDX(ctx, rt, "MLIR_BuildBinary2Attr");
    rv->def = op;
    rv->index = VAL_TRACKED;
    rv->first_use = 0;
    rv->n_uses = 0;

    IR_Operand *rec = OPNDS(ctx);
    link_operand_tracked(rec, vals, u0, VAL_IDX(ctx, x, "MLIR_BuildBinary2Attr"), op);
    link_operand_tracked(rec, vals, u0 + 1, VAL_IDX(ctx, y, "MLIR_BuildBinary2Attr"), op | (1u << OPND_IDX_SHIFT));

    uint32_t *pa = EXTRA(ctx) + fa;
    pa[0] = k0; pa[1] = ATTR_IDX(ctx, v0, "MLIR_BuildBinary2Attr");
    pa[2] = k1; pa[3] = ATTR_IDX(ctx, v1, "MLIR_BuildBinary2Attr");
    return VAL_H(ctx, res);
}

void MLIR_SetOpTrailingComment(MLIR_Context *ctx, MLIR_OpHandle oh, string comment) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_SetOpTrailingComment");
    CHECK_OP(ctx, o, "MLIR_SetOpTrailingComment");
    if (comment.size == 0 && !(OPS(ctx)[o].flags & OPF_HAS_AUX)) return;
    uint32_t a = op_aux_get(ctx, o);
    OPAUX(ctx)[a].trailing_comment = comment;
}

void MLIR_SetValueRegisterName(MLIR_Context *ctx, MLIR_ValueHandle vh, string name) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_SetValueRegisterName");
    API_CHECK(v != 0, "MLIR_SetValueRegisterName", "value handle is NULL");
    if (name.size == 0) return;
    uint32_t n = (VALS(ctx)[v].index & VAL_HAS_NAME) ? u32map_get(&ctx->vname_map, v) : 0;
    if (!n) {
        n = vec_push(&ctx->names, 1);
        VALS(ctx)[v].index |= VAL_HAS_NAME;
        u32map_put(&ctx->vname_map, v, n);
    }
    NAMES(ctx)[n] = name;
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
    for (size_t i = idx; i < blk->n_ops; i++) OPS(ctx)[blk->ops[i]].block_pos = (uint32_t)i;
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
    BLOCKS(ctx)[b].region_pos = reg->n_blocks;
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
    return BLOCK_H(ctx, REGIONS(ctx)[r].blocks[idx]);
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
    return OP_H(ctx, BLOCKS(ctx)[b].ops[idx]);
}

MLIR_OpType MLIR_GetOpType(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpType");
    CHECK_OP(ctx, o, "MLIR_GetOpType");
    return (MLIR_OpType)OPS(ctx)[o].op_type;
}

MLIR_OpHandle MLIR_RenameOp(MLIR_Context *ctx, MLIR_OpHandle oh, MLIR_OpType new_type) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_RenameOp");
    CHECK_OP(ctx, o, "MLIR_RenameOp");
    const IR_OpExtra *ox = op_extra_get(ctx, o);
    API_CHECK(!ox || ox->n_regions == 0, "MLIR_RenameOp", "cannot rename op with regions");
    OPS(ctx)[o].op_type = (uint16_t)new_type;
    return oh;
}

size_t MLIR_GetOpNumRegions(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpNumRegions");
    CHECK_OP(ctx, o, "MLIR_GetOpNumRegions");
    const IR_OpExtra *ox = op_extra_get(ctx, o);
    return ox ? ox->n_regions : 0;
}

MLIR_RegionHandle MLIR_GetOpRegion(MLIR_Context *ctx, MLIR_OpHandle oh, size_t idx) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpRegion");
    CHECK_OP(ctx, o, "MLIR_GetOpRegion");
    const IR_OpExtra *ox = op_extra_get(ctx, o);
    CHECK_INDEX(idx, ox ? ox->n_regions : 0, "MLIR_GetOpRegion");
    return REGION_H(ctx, EXTRA(ctx)[ox->first_extra + idx]);
}

string MLIR_GetOpTrailingComment(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpTrailingComment");
    CHECK_OP(ctx, o, "MLIR_GetOpTrailingComment");
    uint32_t a = op_aux_lookup(ctx, o);
    return a ? OPAUX(ctx)[a].trailing_comment : str_lit("");
}

size_t MLIR_GetOpNumAttributes(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpNumAttributes");
    CHECK_OP(ctx, o, "MLIR_GetOpNumAttributes");
    return OPS(ctx)[o].n_attributes;
}

MLIR_AttrKey MLIR_GetOpAttributeKey(MLIR_Context *ctx, MLIR_OpHandle oh, size_t idx) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpAttributeKey");
    CHECK_OP(ctx, o, "MLIR_GetOpAttributeKey");
    CHECK_INDEX(idx, OPS(ctx)[o].n_attributes, "MLIR_GetOpAttributeKey");
    return EXTRA(ctx)[OPS(ctx)[o].first_attr + 2 * idx];
}

MLIR_AttributeHandle MLIR_GetOpAttribute(MLIR_Context *ctx, MLIR_OpHandle oh, size_t idx) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpAttribute");
    CHECK_OP(ctx, o, "MLIR_GetOpAttribute");
    CHECK_INDEX(idx, OPS(ctx)[o].n_attributes, "MLIR_GetOpAttribute");
    return ATTR_H(ctx, EXTRA(ctx)[OPS(ctx)[o].first_attr + 2 * idx + 1]);
}

// A scan of the op's (key, value) pairs comparing keys.
MLIR_AttributeHandle MLIR_GetOpAttr(MLIR_Context *ctx, MLIR_OpHandle oh, MLIR_AttrKey key) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpAttr");
    CHECK_OP(ctx, o, "MLIR_GetOpAttr");
    const IR_Op *op = &OPS(ctx)[o];
    uint32_t n = op->n_attributes;
    if (n == 0) return MLIR_INVALID_HANDLE;
    const uint32_t *p = EXTRA(ctx) + op->first_attr;
    const uint32_t *end = p + 2 * (size_t)n;
    for (; p < end; p += 2)
        if (p[0] == key) return ATTR_H(ctx, p[1]);
    return MLIR_INVALID_HANDLE;
}

MLIR_AttrKey MLIR_InternAttrKey(MLIR_Context *ctx, string name) {
    API_CHECK(ctx != NULL, "MLIR_InternAttrKey", "context is NULL");
    API_CHECK(name.size > 0, "MLIR_InternAttrKey", "empty attribute name");
    return attr_key_table_intern(&ctx->keys, name);
}

MLIR_AttrKey MLIR_LookupAttrKey(MLIR_Context *ctx, string name) {
    API_CHECK(ctx != NULL, "MLIR_LookupAttrKey", "context is NULL");
    return attr_key_table_lookup(&ctx->keys, name);
}

string MLIR_GetAttrKeyName(MLIR_Context *ctx, MLIR_AttrKey key) {
    API_CHECK(ctx != NULL, "MLIR_GetAttrKeyName", "context is NULL");
    return attr_key_table_name(&ctx->keys, key);
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
    return VAL_H(ctx, OPNDS(ctx)[OPS(ctx)[o].first_operand + idx].value);
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
    return VAL_H(ctx, OPS(ctx)[o].first_result + (uint32_t)idx);
}

MLIR_LocationHandle MLIR_GetOpLocation(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpLocation");
    CHECK_OP(ctx, o, "MLIR_GetOpLocation");
    return LOC_H(ctx, OPS(ctx)[o].location);
}

size_t MLIR_GetOpNumResultTypes(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpNumResultTypes");
    CHECK_OP(ctx, o, "MLIR_GetOpNumResultTypes");
    uint32_t a = op_aux_lookup(ctx, o);
    if (a && OPAUX(ctx)[a].explicit_result_types) return OPAUX(ctx)[a].n_result_types;
    return OPS(ctx)[o].n_results;
}

MLIR_TypeHandle MLIR_GetOpResultType(MLIR_Context *ctx, MLIR_OpHandle oh, size_t idx) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpResultType");
    CHECK_OP(ctx, o, "MLIR_GetOpResultType");
    CHECK_INDEX(idx, MLIR_GetOpNumResultTypes(ctx, oh), "MLIR_GetOpResultType");
    uint32_t a = op_aux_lookup(ctx, o);
    if (a && OPAUX(ctx)[a].explicit_result_types)
        return TYPE_H(ctx, EXTRA(ctx)[OPAUX(ctx)[a].first_result_type + idx]);
    return TYPE_H(ctx, VALS(ctx)[OPS(ctx)[o].first_result + idx].type);
}

size_t MLIR_GetOpNumSuccessors(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpNumSuccessors");
    CHECK_OP(ctx, o, "MLIR_GetOpNumSuccessors");
    const IR_OpExtra *ox = op_extra_get(ctx, o);
    return ox ? ox->n_successors : 0;
}

// The (block, operand count, pred edge) triple of successor `s` of op `o`.
static inline uint32_t *op_succ_pair(MLIR_Context *ctx, uint32_t o, size_t s) {
    const IR_OpExtra *ox = op_extra_get(ctx, o);
    return EXTRA(ctx) + ox->first_extra + ox->n_regions + 3 * s;
}

// The flat operand position (from the op's first operand) of successor
// operand (s, 0).
static uint32_t op_succ_operand_base(MLIR_Context *ctx, uint32_t o, size_t s) {
    IR_Op *op = &OPS(ctx)[o];
    const IR_OpExtra *ox = op_extra_get(ctx, o);
    const uint32_t *triples = EXTRA(ctx) + ox->first_extra + ox->n_regions;
    uint32_t off = op->n_operands;
    for (size_t k = 0; k < s; k++) off += triples[3 * k + 1];
    return off;
}

MLIR_BlockHandle MLIR_GetOpSuccessor(MLIR_Context *ctx, MLIR_OpHandle oh, size_t idx) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpSuccessor");
    CHECK_OP(ctx, o, "MLIR_GetOpSuccessor");
    CHECK_INDEX(idx, MLIR_GetOpNumSuccessors(ctx, oh), "MLIR_GetOpSuccessor");
    return BLOCK_H(ctx, op_succ_pair(ctx, o, idx)[0]);
}

size_t MLIR_GetOpNumSuccessorOperands(MLIR_Context *ctx, MLIR_OpHandle oh, size_t s) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpNumSuccessorOperands");
    CHECK_OP(ctx, o, "MLIR_GetOpNumSuccessorOperands");
    CHECK_INDEX(s, MLIR_GetOpNumSuccessors(ctx, oh), "MLIR_GetOpNumSuccessorOperands");
    return op_succ_pair(ctx, o, s)[1];
}

MLIR_ValueHandle MLIR_GetOpSuccessorOperand(MLIR_Context *ctx, MLIR_OpHandle oh, size_t s, size_t i) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpSuccessorOperand");
    CHECK_OP(ctx, o, "MLIR_GetOpSuccessorOperand");
    CHECK_INDEX(s, MLIR_GetOpNumSuccessors(ctx, oh), "MLIR_GetOpSuccessorOperand");
    CHECK_INDEX(i, op_succ_pair(ctx, o, s)[1], "MLIR_GetOpSuccessorOperand");
    uint32_t u = OPS(ctx)[o].first_operand + op_succ_operand_base(ctx, o, s) + (uint32_t)i;
    return VAL_H(ctx, OPNDS(ctx)[u].value);
}

void MLIR_GetOpSuccessorOperandRange(MLIR_Context *ctx, MLIR_OpHandle oh,
                                     size_t succ, size_t *first, size_t *count) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpSuccessorOperandRange");
    CHECK_OP(ctx, o, "MLIR_GetOpSuccessorOperandRange");
    CHECK_INDEX(succ, MLIR_GetOpNumSuccessors(ctx, oh), "MLIR_GetOpSuccessorOperandRange");
    if (first) *first = op_succ_operand_base(ctx, o, succ);
    if (count) *count = op_succ_pair(ctx, o, succ)[1];
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
    return VAL_H(ctx, BLOCKS(ctx)[b].args[idx]);
}

size_t MLIR_GetBlockIndex(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockIndex");
    if (!b) return (size_t)-1;
    uint32_t r = BLOCKS(ctx)[b].parent_region;
    if (!r) return (size_t)-1;
    return (size_t)BLOCKS(ctx)[b].region_pos;
}

MLIR_LocationHandle MLIR_GetValueLocation(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_GetValueLocation");
    if (!v) return MLIR_INVALID_HANDLE;
    IR_Value *p = &VALS(ctx)[v];
    if (p->index & VAL_HAS_LOC) return LOC_H(ctx, u32map_get(&ctx->vloc_map, v));
    if (!(p->index & VAL_BLOCK_ARG) && p->def != 0)
        return LOC_H(ctx, OPS(ctx)[p->def].location);
    return MLIR_INVALID_HANDLE;
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

// Dimension `i` of a shaped type (two child words each; -1 is dynamic).
static int64_t type_dim(MLIR_Context *c, const IR_Type *t, uint32_t i) {
    const uint32_t *w = TYX(c) + t->first + 2 * i;
    return (int64_t)(((uint64_t)w[1] << 32) | w[0]);
}

// `tensor<2x?x{elem}>` and the like.
static string shaped_type_string(MLIR_Context *ctx, const IR_Type *t, const char *kw) {
    Arena *arena = ctx->arena;
    if (t->elem == 0) return format(arena, str_lit("{}<?>"), str_from_cstr_view((char *)kw));
    string elem_str = MLIR_GetTypeString(ctx, TYPE_H(ctx, t->elem));
    strbuf shape_str = strbuf_make();
    for (uint32_t i = 0; i < t->n / 2; i++) {
        int64_t dim = type_dim(ctx, t, i);
        if (dim < 0) strbuf_append(arena, &shape_str, str_lit("?x"));
        else strbuf_append(arena, &shape_str, format(arena, str_lit("{}x"), dim));
    }
    return format(arena, str_lit("{}<{}{}>"), str_from_cstr_view((char *)kw),
                  strbuf_to_string(shape_str), elem_str);
}

// `T0, T1, ...` for the child words [from, to) of a type.
static string type_list_string(MLIR_Context *ctx, uint32_t first, uint32_t from, uint32_t to) {
    Arena *arena = ctx->arena;
    strbuf sb = strbuf_make();
    for (uint32_t i = from; i < to; i++) {
        if (i > from) strbuf_append(arena, &sb, str_lit(", "));
        strbuf_append(arena, &sb, MLIR_GetTypeString(ctx, TYPE_H(ctx, TYX(ctx)[first + i])));
    }
    return strbuf_to_string(sb);
}

string MLIR_GetTypeString(MLIR_Context *ctx, MLIR_TypeHandle th) {
    if (th == MLIR_INVALID_HANDLE) return str_lit("null");
    if (!ctx || !ctx->arena) return str_lit("null");
    Arena *arena = ctx->arena;
    // A copy: printing allocates in the arena only, but the record is small.
    IR_Type t = TYPES(ctx)[TYPE_IDX(ctx, th, "MLIR_GetTypeString")];
    const uint32_t *kids = TYX(ctx) + t.first;

    switch ((MLIR_TypeKind)t.kind) {
        case MLIR_TYPE_KIND_NONE:
            return str_lit("?");
        case MLIR_TYPE_KIND_OPAQUE:
            if (t.flags & TYF_TT_PTR) {
                if (kids[0] == 0) return str_lit("!tt.ptr<?>");
                string elem_str = MLIR_GetTypeString(ctx, TYPE_H(ctx, kids[0]));
                if (t.flags & TYF_HAS_AS)
                    return format(arena, str_lit("!tt.ptr<{}, {}>"), elem_str, (int64_t)t.aux);
                return format(arena, str_lit("!tt.ptr<{}>"), elem_str);
            }
            return str_lit("unknown");
        case MLIR_TYPE_KIND_INTEGER:
            // Signless `i{w}`, as upstream prints MLIR_CreateTypeInteger's types.
            return format(arena, str_lit("i{}"), (int64_t)t.width);
        case MLIR_TYPE_KIND_FLOAT:
            return format(arena, str_lit("f{}"), (int64_t)t.width);
        case MLIR_TYPE_KIND_BF16:
            return str_lit("bf16");
        case MLIR_TYPE_KIND_INDEX:
            return str_lit("index");
        case MLIR_TYPE_KIND_TENSOR:
            return shaped_type_string(ctx, &t, "tensor");
        case MLIR_TYPE_KIND_MEMREF:
            return shaped_type_string(ctx, &t, "memref");
        case MLIR_TYPE_KIND_VECTOR:
            return shaped_type_string(ctx, &t, "vector");
        case MLIR_TYPE_KIND_FUNCTION: {
            uint32_t n_in = t.n - t.aux;
            string in_str = type_list_string(ctx, t.first, 0, n_in);
            string out_str = t.aux == 1
                ? MLIR_GetTypeString(ctx, TYPE_H(ctx, kids[n_in]))
                : format(arena, str_lit("({})"), type_list_string(ctx, t.first, n_in, t.n));
            return format(arena, str_lit("({}) -> {}"), in_str, out_str);
        }
        case MLIR_TYPE_KIND_LLVM_POINTER:
            if (t.aux == 0) return str_lit("!llvm.ptr");
            return format(arena, str_lit("!llvm.ptr<{}>"), (int64_t)t.aux);
        case MLIR_TYPE_KIND_LLVM_VOID:
            return str_lit("!llvm.void");
        case MLIR_TYPE_KIND_LLVM_ARRAY:
            return format(arena, str_lit("!llvm.array<{} x {}>"), (int64_t)t.count,
                          MLIR_GetTypeString(ctx, TYPE_H(ctx, t.elem)));
        case MLIR_TYPE_KIND_LLVM_STRUCT: {
            // Literal: !llvm.struct<(T1, T2)>; identified: !llvm.struct<"name", (T1, T2)>.
            // Nested struct fields keep the "!llvm." prefix (upstream drops it);
            // the LLVM-IR translator accepts both.
            // An identified struct whose body was never set is opaque, as
            // upstream prints it.
            string body = (t.flags & TYF_IDENTIFIED) && !(t.flags & TYF_BODY_SET)
                ? str_lit("opaque")
                : format(arena, str_lit("({})"), type_list_string(ctx, t.first, 0, t.n));
            if ((t.flags & TYF_IDENTIFIED) && TNAMES(ctx)[t.aux].size > 0)
                return format(arena, str_lit("!llvm.struct<\"{}\", {}>"), TNAMES(ctx)[t.aux], body);
            return format(arena, str_lit("!llvm.struct<{}>"), body);
        }
        case MLIR_TYPE_KIND_LLVM_FUNCTION: {
            string ret = MLIR_GetTypeString(ctx, TYPE_H(ctx, kids[0]));
            string params = type_list_string(ctx, t.first, 1, t.n);
            if (t.flags & TYF_VAR_ARG)
                params = format(arena, str_lit("{}{}..."), params,
                                t.n > 1 ? str_lit(", ") : str_lit(""));
            return format(arena, str_lit("!llvm.func<{} ({})>"), ret, params);
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
    // backends produce the same canonical type identity for `i{w}`.
    (void)is_signed;
    if (!ctx) return MLIR_INVALID_HANDLE;
    if (width <= 64 && ctx->int_types[width]) return TYPE_H(ctx, ctx->int_types[width]);
    IR_Type t = type_make(MLIR_TYPE_KIND_INTEGER);
    t.width = width;
    uint32_t i = type_intern(ctx, t, (string){0});
    if (width <= 64) ctx->int_types[width] = i;
    return TYPE_H(ctx, i);
}

// bf16 for `is_bfloat` (whatever the width), else f16, f32 or f64; any other
// width is f32, as upstream.
MLIR_TypeHandle MLIR_CreateTypeFloat(MLIR_Context *ctx, uint32_t width, bool is_bfloat) {
    if (!ctx) return MLIR_INVALID_HANDLE;
    IR_Type t = type_make(is_bfloat ? MLIR_TYPE_KIND_BF16 : MLIR_TYPE_KIND_FLOAT);
    t.width = is_bfloat ? 16 : (width == 16 || width == 64) ? width : 32;
    return TYPE_H(ctx, type_intern(ctx, t, (string){0}));
}

static MLIR_TypeHandle create_simple_type(MLIR_Context *ctx, MLIR_TypeKind k) {
    if (!ctx) return MLIR_INVALID_HANDLE;
    return TYPE_H(ctx, type_intern(ctx, type_make(k), (string){0}));
}

MLIR_TypeHandle MLIR_CreateTypeIndex(MLIR_Context *ctx) {
    return create_simple_type(ctx, MLIR_TYPE_KIND_INDEX);
}

MLIR_TypeHandle MLIR_CreateTypeUnknown(MLIR_Context *ctx) {
    return create_simple_type(ctx, MLIR_TYPE_KIND_NONE);
}

// Interned by name, as upstream keys opaque types by their data. (It prints
// as `unknown`, the only opaque type the parser makes.)
MLIR_TypeHandle MLIR_CreateTypeOpaque(MLIR_Context *ctx, string name) {
    if (!ctx) return MLIR_INVALID_HANDLE;
    IR_Type t = type_make(MLIR_TYPE_KIND_OPAQUE);
    t.flags = TYF_IDENTIFIED;
    return TYPE_H(ctx, type_intern(ctx, t, name));
}

MLIR_TypeHandle MLIR_CreateTypeLLVMVoid(MLIR_Context *ctx) {
    return create_simple_type(ctx, MLIR_TYPE_KIND_LLVM_VOID);
}

// Tensor, memref and vector types: the dimensions are two child words each.
// No shape (NULL) is rank 0, as upstream. A vector's element count is the
// product of its dimensions.
static MLIR_TypeHandle create_shaped_type(MLIR_Context *ctx, MLIR_TypeKind k, const int64_t *shape,
                                          size_t rank, MLIR_TypeHandle element_type, const char *fn) {
    if (!ctx) return MLIR_INVALID_HANDLE;
    if (!shape) rank = 0;
    IR_Type t = type_make(k);
    t.elem = TYPE_IDX(ctx, element_type, fn);
    t.n = (uint32_t)(2 * rank);
    t.first = vec_push(&ctx->tyx, t.n);
    uint64_t count = 1;
    for (size_t i = 0; i < rank; i++) {
        // Every negative size is the one dynamic size, as upstream.
        uint64_t d = (uint64_t)(shape[i] < 0 ? -1 : shape[i]);
        TYX(ctx)[t.first + 2 * i] = (uint32_t)d;
        TYX(ctx)[t.first + 2 * i + 1] = (uint32_t)(d >> 32);
        count *= d;
    }
    if (k == MLIR_TYPE_KIND_VECTOR) t.count = count;
    return TYPE_H(ctx, type_intern(ctx, t, (string){0}));
}

MLIR_TypeHandle MLIR_CreateTypeTensor(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type) {
    return create_shaped_type(ctx, MLIR_TYPE_KIND_TENSOR, shape, rank, element_type, "MLIR_CreateTypeTensor");
}

MLIR_TypeHandle MLIR_CreateTypeMemref(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type) {
    return create_shaped_type(ctx, MLIR_TYPE_KIND_MEMREF, shape, rank, element_type, "MLIR_CreateTypeMemref");
}

MLIR_TypeHandle MLIR_CreateTypeVector(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type) {
    API_CHECK(shape != NULL && rank > 0, "MLIR_CreateTypeVector", "a vector needs a shape");
    API_CHECK(element_type != MLIR_INVALID_HANDLE, "MLIR_CreateTypeVector", "element type is NULL");
#ifdef MLIR_API_CHECKS
    for (size_t i = 0; i < rank; i++)
        API_CHECK(shape[i] > 0, "MLIR_CreateTypeVector", "a vector dimension must be static and positive");
#endif
    return create_shaped_type(ctx, MLIR_TYPE_KIND_VECTOR, shape, rank, element_type, "MLIR_CreateTypeVector");
}

// The Triton test type !tt.ptr<pointee[, address space]>: an OPAQUE type to
// the API, as on upstream.
MLIR_TypeHandle MLIR_CreateTypePointer(MLIR_Context *ctx, MLIR_TypeHandle element_type, bool has_address_space, uint32_t address_space) {
    if (!ctx) return MLIR_INVALID_HANDLE;
    IR_Type t = type_make(MLIR_TYPE_KIND_OPAQUE);
    t.flags = (uint8_t)(TYF_TT_PTR | (has_address_space ? TYF_HAS_AS : 0));
    t.aux = has_address_space ? address_space : 0;
    t.n = 1;
    t.first = tyx_push_types(ctx, &element_type, 1, "MLIR_CreateTypePointer");
    return TYPE_H(ctx, type_intern(ctx, t, (string){0}));
}

MLIR_TypeHandle MLIR_CreateTypeLLVMPointer(MLIR_Context *ctx) {
    return MLIR_CreateTypeLLVMPointerInAddressSpace(ctx, 0);
}

MLIR_TypeHandle MLIR_CreateTypeLLVMPointerInAddressSpace(
    MLIR_Context *ctx, uint32_t address_space) {
    if (!ctx) return MLIR_INVALID_HANDLE;
    if (address_space == 0 && ctx->ptr_type) return TYPE_H(ctx, ctx->ptr_type);
    IR_Type t = type_make(MLIR_TYPE_KIND_LLVM_POINTER);
    t.aux = address_space;
    uint32_t i = type_intern(ctx, t, (string){0});
    if (address_space == 0) ctx->ptr_type = i;
    return TYPE_H(ctx, i);
}

// Identified LLVM structs are interned by name within a context: the same
// printed reference (`!llvm.struct<"foo", ...>`) must resolve to the same
// type handle across all uses, and two structs with the same body but
// different names are different types. The body is set later, once.
MLIR_TypeHandle MLIR_CreateTypeLLVMStructIdentified(MLIR_Context *ctx, string name) {
    if (!ctx) return MLIR_INVALID_HANDLE;
    IR_Type t = type_make(MLIR_TYPE_KIND_LLVM_STRUCT);
    t.flags = TYF_IDENTIFIED;
    return TYPE_H(ctx, type_intern(ctx, t, name));
}

void MLIR_SetTypeLLVMStructBody(MLIR_Context *ctx, MLIR_TypeHandle struct_ty,
                                 const MLIR_TypeHandle *fields, size_t n_fields) {
    uint32_t i = TYPE_IDX(ctx, struct_ty, "MLIR_SetTypeLLVMStructBody");
    if (i == 0 || TYPES(ctx)[i].kind != MLIR_TYPE_KIND_LLVM_STRUCT) return;
    if (TYPES(ctx)[i].flags & TYF_BODY_SET) return; // upstream silently ignores re-set
    // Push the fields first, then resolve the record (the pointer rule).
    uint32_t first = tyx_push_types(ctx, fields, n_fields, "MLIR_SetTypeLLVMStructBody");
    IR_Type *t = &TYPES(ctx)[i];
    t->first = first;
    t->n = (uint32_t)n_fields;
    t->flags |= TYF_BODY_SET;
}

MLIR_TypeHandle MLIR_CreateTypeLLVMArray(MLIR_Context *ctx, MLIR_TypeHandle elem, uint64_t count) {
    if (!ctx) return MLIR_INVALID_HANDLE;
    IR_Type t = type_make(MLIR_TYPE_KIND_LLVM_ARRAY);
    t.elem = TYPE_IDX(ctx, elem, "MLIR_CreateTypeLLVMArray");
    t.count = count;
    return TYPE_H(ctx, type_intern(ctx, t, (string){0}));
}

// Child words: the result (the LLVM void type for none), then the inputs.
MLIR_TypeHandle MLIR_CreateTypeLLVMFunction(MLIR_Context *ctx,
                                             MLIR_TypeHandle result,
                                             const MLIR_TypeHandle *inputs,
                                             size_t n_inputs,
                                             bool is_var_arg) {
    if (!ctx) return MLIR_INVALID_HANDLE;
    IR_Type t = type_make(MLIR_TYPE_KIND_LLVM_FUNCTION);
    t.flags = is_var_arg ? TYF_VAR_ARG : 0;
    if (!inputs) n_inputs = 0;
    t.n = (uint32_t)(1 + n_inputs);
    t.first = tyx_push_types(ctx, &result, 1, "MLIR_CreateTypeLLVMFunction");
    tyx_push_types(ctx, inputs, n_inputs, "MLIR_CreateTypeLLVMFunction");
    return TYPE_H(ctx, type_intern(ctx, t, (string){0}));
}

// Child words: the inputs, then the results (`aux` of them).
MLIR_TypeHandle MLIR_CreateTypeFunction(MLIR_Context *ctx,
                                         const MLIR_TypeHandle *inputs, size_t n_inputs,
                                         const MLIR_TypeHandle *results, size_t n_results) {
    if (!ctx) return MLIR_INVALID_HANDLE;
    if (!inputs) n_inputs = 0;
    if (!results) n_results = 0;
    IR_Type t = type_make(MLIR_TYPE_KIND_FUNCTION);
    t.aux = (uint32_t)n_results;
    t.n = (uint32_t)(n_inputs + n_results);
    t.first = tyx_push_types(ctx, inputs, n_inputs, "MLIR_CreateTypeFunction");
    tyx_push_types(ctx, results, n_results, "MLIR_CreateTypeFunction");
    return TYPE_H(ctx, type_intern(ctx, t, (string){0}));
}

// Type introspection: one read of the type's record each.
#define TYPE_REC(ctx, th, fn) (&TYPES(ctx)[TYPE_IDX((ctx), (th), (fn))])

MLIR_TypeKind MLIR_GetTypeKind(MLIR_Context *ctx, MLIR_TypeHandle th) {
    return (MLIR_TypeKind)TYPE_REC(ctx, th, "MLIR_GetTypeKind")->kind;
}

uint32_t MLIR_GetTypeWidth(MLIR_Context *ctx, MLIR_TypeHandle th) {
    return TYPE_REC(ctx, th, "MLIR_GetTypeWidth")->width;
}

MLIR_TypeHandle MLIR_GetTypeElement(MLIR_Context *ctx, MLIR_TypeHandle th) {
    return TYPE_H(ctx, TYPE_REC(ctx, th, "MLIR_GetTypeElement")->elem);
}

uint64_t MLIR_GetTypeNumElements(MLIR_Context *ctx, MLIR_TypeHandle th) {
    return TYPE_REC(ctx, th, "MLIR_GetTypeNumElements")->count;
}

static bool type_kind_is_shaped(uint32_t k) {
    return k == MLIR_TYPE_KIND_TENSOR || k == MLIR_TYPE_KIND_MEMREF || k == MLIR_TYPE_KIND_VECTOR;
}

size_t MLIR_GetTypeShapeRank(MLIR_Context *ctx, MLIR_TypeHandle th) {
    const IR_Type *t = TYPE_REC(ctx, th, "MLIR_GetTypeShapeRank");
    return type_kind_is_shaped(t->kind) ? t->n / 2 : 0;
}

int64_t MLIR_GetTypeShapeDim(MLIR_Context *ctx, MLIR_TypeHandle th, size_t idx) {
    const IR_Type *t = TYPE_REC(ctx, th, "MLIR_GetTypeShapeDim");
    CHECK_INDEX(idx, type_kind_is_shaped(t->kind) ? t->n / 2 : 0, "MLIR_GetTypeShapeDim");
    return type_dim(ctx, t, (uint32_t)idx);
}

size_t MLIR_GetTypeFunctionNumInputs(MLIR_Context *ctx, MLIR_TypeHandle th) {
    const IR_Type *t = TYPE_REC(ctx, th, "MLIR_GetTypeFunctionNumInputs");
    if (t->kind == MLIR_TYPE_KIND_FUNCTION) return t->n - t->aux;
    if (t->kind == MLIR_TYPE_KIND_LLVM_FUNCTION) return t->n - 1;
    return 0;
}

MLIR_TypeHandle MLIR_GetTypeFunctionInput(MLIR_Context *ctx, MLIR_TypeHandle th, size_t idx) {
    size_t n = MLIR_GetTypeFunctionNumInputs(ctx, th);
    API_CHECK(idx < n, "MLIR_GetTypeFunctionInput", "index out of range");
    if (idx >= n) return MLIR_INVALID_HANDLE;
    const IR_Type *t = TYPE_REC(ctx, th, "MLIR_GetTypeFunctionInput");
    uint32_t skip = t->kind == MLIR_TYPE_KIND_LLVM_FUNCTION ? 1 : 0;
    return TYPE_H(ctx, TYX(ctx)[t->first + skip + idx]);
}

// An LLVM function type returning void has no results, as a builtin
// function type with none.
size_t MLIR_GetTypeFunctionNumResults(MLIR_Context *ctx, MLIR_TypeHandle th) {
    const IR_Type *t = TYPE_REC(ctx, th, "MLIR_GetTypeFunctionNumResults");
    if (t->kind == MLIR_TYPE_KIND_FUNCTION) return t->aux;
    if (t->kind == MLIR_TYPE_KIND_LLVM_FUNCTION)
        return TYPES(ctx)[TYX(ctx)[t->first]].kind == MLIR_TYPE_KIND_LLVM_VOID ? 0 : 1;
    return 0;
}

MLIR_TypeHandle MLIR_GetTypeFunctionResult(MLIR_Context *ctx, MLIR_TypeHandle th, size_t idx) {
    size_t n = MLIR_GetTypeFunctionNumResults(ctx, th);
    API_CHECK(idx < n, "MLIR_GetTypeFunctionResult", "index out of range");
    if (idx >= n) return MLIR_INVALID_HANDLE;
    const IR_Type *t = TYPE_REC(ctx, th, "MLIR_GetTypeFunctionResult");
    if (t->kind == MLIR_TYPE_KIND_LLVM_FUNCTION) return TYPE_H(ctx, TYX(ctx)[t->first]);
    return TYPE_H(ctx, TYX(ctx)[t->first + (t->n - t->aux) + idx]);
}

bool MLIR_GetTypeFunctionIsVarArg(MLIR_Context *ctx, MLIR_TypeHandle th) {
    const IR_Type *t = TYPE_REC(ctx, th, "MLIR_GetTypeFunctionIsVarArg");
    return t->kind == MLIR_TYPE_KIND_LLVM_FUNCTION && (t->flags & TYF_VAR_ARG);
}

size_t MLIR_GetTypeLLVMStructNumFields(MLIR_Context *ctx, MLIR_TypeHandle th) {
    const IR_Type *t = TYPE_REC(ctx, th, "MLIR_GetTypeLLVMStructNumFields");
    return t->kind == MLIR_TYPE_KIND_LLVM_STRUCT ? t->n : 0;
}

MLIR_TypeHandle MLIR_GetTypeLLVMStructField(MLIR_Context *ctx, MLIR_TypeHandle th, size_t idx) {
    size_t n = MLIR_GetTypeLLVMStructNumFields(ctx, th);
    API_CHECK(idx < n, "MLIR_GetTypeLLVMStructField", "index out of range");
    if (idx >= n) return MLIR_INVALID_HANDLE;
    return TYPE_H(ctx, TYX(ctx)[TYPE_REC(ctx, th, "MLIR_GetTypeLLVMStructField")->first + idx]);
}

string MLIR_GetTypeLLVMStructName(MLIR_Context *ctx, MLIR_TypeHandle th) {
    const IR_Type *t = TYPE_REC(ctx, th, "MLIR_GetTypeLLVMStructName");
    if (t->kind != MLIR_TYPE_KIND_LLVM_STRUCT || !(t->flags & TYF_IDENTIFIED)) return str_lit("");
    return TNAMES(ctx)[t->aux];
}

bool MLIR_IsTypeLLVMStructOpaque(MLIR_Context *ctx, MLIR_TypeHandle th) {
    const IR_Type *t = TYPE_REC(ctx, th, "MLIR_IsTypeLLVMStructOpaque");
    return t->kind == MLIR_TYPE_KIND_LLVM_STRUCT && (t->flags & TYF_IDENTIFIED) &&
           !(t->flags & TYF_BODY_SET);
}

uint32_t MLIR_GetTypeLLVMPointerAddressSpace(MLIR_Context *ctx, MLIR_TypeHandle th) {
    const IR_Type *t = TYPE_REC(ctx, th, "MLIR_GetTypeLLVMPointerAddressSpace");
    return t->kind == MLIR_TYPE_KIND_LLVM_POINTER ? t->aux : 0;
}

// LLVM-dialect global helpers. Construct `llvm.mlir.global` ops with the
// attributes the LLVM-IR translator looks for (sym_name, global_type,
// linkage, constant, value). Both helpers return the op so the caller
// can append it to the module body. MLIR_CreateLLVMGlobal optionally
// hands back the entry block of the initializer region for region-init
// globals.
static MLIR_AttributeHandle make_llvm_linkage_attr(MLIR_Context *ctx, string linkage_kind) {
    // Mirror the upstream printer: "#llvm.linkage<<kind>>".
    char buf[64];
    size_t n = 0;
    const char *pre = "#llvm.linkage<";
    for (const char *p = pre; *p; p++) buf[n++] = *p;
    for (size_t i = 0; i < linkage_kind.size && n < sizeof buf - 1; i++) buf[n++] = linkage_kind.str[i];
    buf[n++] = '>';
    string text = { buf, n };
    return MLIR_CreateAttributeString(ctx, text);
}

MLIR_OpHandle MLIR_CreateLLVMGlobalString(MLIR_Context *ctx, string sym_name,
                                          string bytes, MLIR_LocationHandle loc) {
    MLIR_TypeHandle i8 = MLIR_CreateTypeInteger(ctx, 8, false);
    MLIR_TypeHandle arr = MLIR_CreateTypeLLVMArray(ctx, i8, bytes.size);
    MLIR_NamedAttr attrs[5];
    attrs[0] = MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, sym_name));
    attrs[1] = MLIR_Named(MLIR_KEY_GLOBAL_TYPE, MLIR_CreateAttributeType(ctx, arr));
    attrs[2] = MLIR_Named(MLIR_KEY_LINKAGE, make_llvm_linkage_attr(ctx, str_lit("private")));
    attrs[3] = MLIR_Named(MLIR_KEY_CONSTANT, MLIR_CreateAttributeBool(ctx, true));
    attrs[4] = MLIR_Named(MLIR_KEY_VALUE, MLIR_CreateAttributeString(ctx, bytes));
    MLIR_OpSpec spec = {0};
    spec.type = OP_TYPE_LLVM_MLIR_GLOBAL;
    spec.attrs = attrs;
    spec.n_attrs = 5;
    spec.loc = loc;
    return MLIR_CreateOpFromSpec(ctx, &spec);
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
    MLIR_NamedAttr attrs[5];
    size_t na = 0;
    attrs[na++] = MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, sym_name));
    attrs[na++] = MLIR_Named(MLIR_KEY_GLOBAL_TYPE, MLIR_CreateAttributeType(ctx, array_ty));
    attrs[na++] = MLIR_Named(MLIR_KEY_LINKAGE, make_llvm_linkage_attr(ctx, str_lit("internal")));
    if (is_constant) {
        attrs[na++] = MLIR_Named(MLIR_KEY_CONSTANT, MLIR_CreateAttributeBool(ctx, true));
    }
    attrs[na++] = MLIR_Named(MLIR_KEY_VALUE, MLIR_CreateAttributeString(ctx, bytes));
    MLIR_OpSpec spec = {0};
    spec.type = OP_TYPE_LLVM_MLIR_GLOBAL;
    spec.attrs = attrs;
    spec.n_attrs = (uint32_t)na;
    spec.loc = loc;
    return MLIR_CreateOpFromSpec(ctx, &spec);
}

MLIR_OpHandle MLIR_CreateLLVMGlobal(MLIR_Context *ctx, string sym_name,
                                    MLIR_TypeHandle elem_ty, bool is_constant,
                                    int init_kind, int64_t init_int, double init_float,
                                    MLIR_BlockHandle *out_init_block,
                                    MLIR_LocationHandle loc) {
    // init_kind: 0=integer, 1=float, 2=region, 3=zero-init/no-init, 4=external
    MLIR_NamedAttr attrs[6];
    size_t na = 0;
    attrs[na++] = MLIR_Named(MLIR_KEY_SYM_NAME, MLIR_CreateAttributeString(ctx, sym_name));
    attrs[na++] = MLIR_Named(MLIR_KEY_GLOBAL_TYPE, MLIR_CreateAttributeType(ctx, elem_ty));
    string lk = (init_kind == 4) ? str_lit("external") : str_lit("internal");
    attrs[na++] = MLIR_Named(MLIR_KEY_LINKAGE, make_llvm_linkage_attr(ctx, lk));
    if (is_constant) attrs[na++] = MLIR_Named(MLIR_KEY_CONSTANT, MLIR_CreateAttributeBool(ctx, true));
    if (init_kind == 0) {
        attrs[na++] = MLIR_Named(MLIR_KEY_VALUE, MLIR_CreateAttributeInteger(ctx, init_int, elem_ty));
    } else if (init_kind == 1) {
        attrs[na++] = MLIR_Named(MLIR_KEY_VALUE, MLIR_CreateAttributeFloat(ctx, init_float, elem_ty));
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
    MLIR_OpSpec spec = {0};
    spec.type = OP_TYPE_LLVM_MLIR_GLOBAL;
    spec.attrs = attrs;
    spec.n_attrs = (uint32_t)na;
    spec.regions = nr ? regs : NULL;
    spec.n_regions = (uint32_t)nr;
    spec.loc = loc;
    return MLIR_CreateOpFromSpec(ctx, &spec);
}

// Attribute creation: build the candidate record and intern it.
MLIR_AttributeHandle MLIR_CreateAttributeInteger(MLIR_Context *ctx, int64_t value, MLIR_TypeHandle type) {
    return ATTR_H(ctx, attr_intern_scalar(ctx, AK_INTEGER, TYPE_IDX(ctx, type, "MLIR_CreateAttributeInteger"),
                                          (uint64_t)value));
}

MLIR_AttributeHandle MLIR_CreateAttributeString(MLIR_Context *ctx, string value) {
    IR_Attr a;
    attr_init(&a, AK_STRING);
    a.n = (uint32_t)value.size;
    a.bits = (uint64_t)(uintptr_t)value.str;
    return ATTR_H(ctx, attr_intern(ctx, &a));
}

// By bit pattern: 0.0 and -0.0 differ, a NaN equals itself.
MLIR_AttributeHandle MLIR_CreateAttributeFloat(MLIR_Context *ctx, double value, MLIR_TypeHandle type) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof bits);
    return ATTR_H(ctx, attr_intern_scalar(ctx, AK_FLOAT, TYPE_IDX(ctx, type, "MLIR_CreateAttributeFloat"), bits));
}

MLIR_AttributeHandle MLIR_CreateAttributeBool(MLIR_Context *ctx, bool value) {
    return ATTR_H(ctx, attr_intern_scalar(ctx, AK_BOOL, 0, value ? 1 : 0));
}

MLIR_AttributeHandle MLIR_CreateAttributeArray(MLIR_Context *ctx, const MLIR_AttributeHandle *elements, size_t count) {
    IR_Attr a;
    attr_init(&a, AK_ARRAY);
    if (!elements) count = 0;
    a.n = (uint32_t)count;
    a.first = vec_push(&ctx->ax, a.n);
    for (size_t i = 0; i < count; i++)
        AX(ctx)[a.first + i] = ATTR_IDX(ctx, elements[i], "MLIR_CreateAttributeArray");
    return ATTR_H(ctx, attr_intern(ctx, &a));
}

MLIR_AttributeHandle MLIR_CreateAttributeDict(MLIR_Context *ctx, const MLIR_NamedAttr *entries, size_t count) {
    IR_Attr a;
    attr_init(&a, AK_DICT);
    if (!entries) count = 0;
    a.n = (uint32_t)count;
    a.first = vec_push(&ctx->ax, 2 * a.n);
    for (size_t i = 0; i < count; i++) {
        API_CHECK(entries[i].key != MLIR_KEY_INVALID && entries[i].key < ctx->keys.n_keys &&
                  entries[i].value != MLIR_INVALID_HANDLE, "MLIR_CreateAttributeDict", "bad entry key or value");
        AX(ctx)[a.first + 2 * i] = entries[i].key;
        AX(ctx)[a.first + 2 * i + 1] = ATTR_IDX(ctx, entries[i].value, "MLIR_CreateAttributeDict");
    }
    return ATTR_H(ctx, attr_intern(ctx, &a));
}

MLIR_AttributeHandle MLIR_CreateAttributeType(MLIR_Context *ctx, MLIR_TypeHandle type) {
    return ATTR_H(ctx, attr_intern_scalar(ctx, AK_TYPE, TYPE_IDX(ctx, type, "MLIR_CreateAttributeType"), 0));
}

// A symbol reference is stored as the string `@name`, which is what
// MLIR_GetAttributeString returns for it on both backends.
MLIR_AttributeHandle MLIR_CreateAttributeSymbolRef(MLIR_Context *ctx, string value) {
    char local[128];
    char *buf = value.size + 1 <= sizeof local ? local : (char *)malloc(value.size + 1);
    buf[0] = '@';
    if (value.size) memcpy(buf + 1, value.str, value.size);
    string s = { buf, value.size + 1 };
    MLIR_AttributeHandle h = MLIR_CreateAttributeString(ctx, s);
    if (buf != local) free(buf);
    return h;
}

// Dense arrays are typed arrays: one word per i32, two per i64.
MLIR_AttributeHandle MLIR_CreateAttributeDenseI32Array(MLIR_Context *ctx,
                                                       const int32_t *values, size_t count) {
    IR_Attr a;
    attr_init(&a, AK_DENSE_I32);
    if (!values) count = 0;
    a.n = (uint32_t)count;
    a.first = vec_push(&ctx->ax, a.n);
    for (size_t i = 0; i < count; i++) AX(ctx)[a.first + i] = (uint32_t)values[i];
    return ATTR_H(ctx, attr_intern(ctx, &a));
}

MLIR_AttributeHandle MLIR_CreateAttributeDenseI64Array(MLIR_Context *ctx,
                                                       const int64_t *values, size_t count) {
    IR_Attr a;
    attr_init(&a, AK_DENSE_I64);
    if (!values) count = 0;
    a.n = (uint32_t)count;
    a.first = vec_push(&ctx->ax, 2 * a.n);
    for (size_t i = 0; i < count; i++) {
        AX(ctx)[a.first + 2 * i] = (uint32_t)(uint64_t)values[i];
        AX(ctx)[a.first + 2 * i + 1] = (uint32_t)((uint64_t)values[i] >> 32);
    }
    return ATTR_H(ctx, attr_intern(ctx, &a));
}

// LLVM linkage attribute. Native backend stores the printable form
// (e.g. "#llvm.linkage<internal>") in a string attr — this matches the
// upstream backend's MLIR_GetAttributeAsString result, so the native
// LLVM-IR translator can recognize it the same way for both backends.
MLIR_AttributeHandle MLIR_CreateAttributeLLVMLinkageInternal(MLIR_Context *ctx) {
    return MLIR_CreateAttributeString(ctx, str_lit("#llvm.linkage<internal>"));
}

// Never interned: a new record each time (`bits` numbers them).
MLIR_AttributeHandle MLIR_CreateAttributeDistinct(MLIR_Context *ctx, MLIR_AttributeHandle referenced) {
    IR_Attr a;
    attr_init(&a, AK_DISTINCT);
    a.type = ATTR_IDX(ctx, referenced, "MLIR_CreateAttributeDistinct");
    uint32_t idx = vec_push(&ctx->attrs, 1);
    a.bits = idx;
    ATTRS(ctx)[idx] = a;
    return ATTR_H(ctx, idx);
}

MLIR_AttributeHandle MLIR_CreateAttributeFastMath(MLIR_Context *ctx, MLIR_FastMath fmf) {
    return ATTR_H(ctx, attr_intern_scalar(ctx, AK_FASTMATH, 0, (uint64_t)(fmf & MLIR_FMF_FAST)));
}

MLIR_FastMath MLIR_GetAttributeFastMath(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    if (ah == MLIR_INVALID_HANDLE) return MLIR_FMF_NONE;
    const IR_Attr *r = ATTR_REC(ctx, ah, "MLIR_GetAttributeFastMath");
    if (r->kind != AK_FASTMATH) return MLIR_FMF_NONE;
    return (MLIR_FastMath)(r->bits & MLIR_FMF_FAST);
}

MLIR_FastMath MLIR_GetOpFastMath(MLIR_Context *ctx, MLIR_OpHandle op) {
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, op, MLIR_KEY_FASTMATH_FLAGS);
    if (a == MLIR_INVALID_HANDLE) return MLIR_FMF_NONE;
    return MLIR_GetAttributeFastMath(ctx, a);
}

void MLIR_SetOpFastMath(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_FastMath fmf) {
    MLIR_SetOpAttr(ctx, op, MLIR_KEY_FASTMATH_FLAGS,
                   MLIR_CreateAttributeFastMath(ctx, fmf));
}

MLIR_AttributeHandle MLIR_CreateAttributeSplat(MLIR_Context *ctx,
                                               MLIR_TypeHandle vec_type,
                                               MLIR_AttributeHandle scalar_attr) {
    uint32_t ti = TYPE_IDX(ctx, vec_type, "MLIR_CreateAttributeSplat");
    uint32_t ai = ATTR_IDX(ctx, scalar_attr, "MLIR_CreateAttributeSplat");
    return ATTR_H(ctx, attr_intern_scalar(ctx, AK_SPLAT, ti, (uint64_t)ai));
}

static bool native_lane_in_bounds(MLIR_Context *ctx, MLIR_TypeHandle ty, int64_t lane) {
    if (lane < 0) return true;
    MLIR_TypeKind k = MLIR_GetTypeKind(ctx, ty);
    if (k == MLIR_TYPE_KIND_VECTOR || k == MLIR_TYPE_KIND_TENSOR) {
        uint64_t n = MLIR_GetTypeNumElements(ctx, ty);
        return (uint64_t)lane < n;
    }
    return lane == 0;
}

bool MLIR_MatchConstantFloat(MLIR_Context *ctx, MLIR_ValueHandle v, int64_t lane, double *out) {
    if (v == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetValueKind(ctx, v) != MLIR_VALUE_OP_RESULT) return false;
    MLIR_OpHandle def = MLIR_GetValueDefiningOp(ctx, v);
    if (def == MLIR_INVALID_HANDLE) return false;
    MLIR_OpType t = MLIR_GetOpType(ctx, def);
    if (t == OP_TYPE_LLVM_MLIR_ZERO) {
        MLIR_TypeHandle vty = MLIR_GetValueType(ctx, v);
        MLIR_TypeHandle ety = (MLIR_GetTypeKind(ctx, vty) == MLIR_TYPE_KIND_VECTOR)
            ? MLIR_GetTypeElement(ctx, vty) : vty;
        MLIR_TypeKind ek = MLIR_GetTypeKind(ctx, ety);
        if ((ek == MLIR_TYPE_KIND_FLOAT || ek == MLIR_TYPE_KIND_BF16) &&
            native_lane_in_bounds(ctx, vty, lane)) {
            if (out) *out = 0.0;
            return true;
        }
        return false;
    }
    if (!MLIR_OpHasTrait(ctx, def, MLIR_T_CONSTANT)) return false;
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, def, MLIR_KEY_VALUE);
    if (a == MLIR_INVALID_HANDLE) return false;
    const IR_Attr *r = ATTR_REC(ctx, a, "MLIR_MatchConstantFloat");
    if (r->kind == AK_FLOAT) {
        if (lane > 0) return false;
        double d;
        memcpy(&d, &r->bits, sizeof d);
        if (out) *out = d;
        return true;
    }
    if (r->kind == AK_SPLAT) {
        MLIR_TypeHandle sty = TYPE_H(ctx, r->type);
        if (!native_lane_in_bounds(ctx, sty, lane)) return false;
        const IR_Attr *sr = &ATTRS(ctx)[(uint32_t)r->bits];
        if (sr->kind != AK_FLOAT) return false;
        double d;
        memcpy(&d, &sr->bits, sizeof d);
        if (out) *out = d;
        return true;
    }
    return false;
}

bool MLIR_MatchConstantInt(MLIR_Context *ctx, MLIR_ValueHandle v, int64_t lane, int64_t *out) {
    if (v == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetValueKind(ctx, v) != MLIR_VALUE_OP_RESULT) return false;
    MLIR_OpHandle def = MLIR_GetValueDefiningOp(ctx, v);
    if (def == MLIR_INVALID_HANDLE) return false;
    MLIR_OpType t = MLIR_GetOpType(ctx, def);
    if (t == OP_TYPE_LLVM_MLIR_ZERO) {
        MLIR_TypeHandle vty = MLIR_GetValueType(ctx, v);
        MLIR_TypeHandle ety = (MLIR_GetTypeKind(ctx, vty) == MLIR_TYPE_KIND_VECTOR)
            ? MLIR_GetTypeElement(ctx, vty) : vty;
        MLIR_TypeKind ek = MLIR_GetTypeKind(ctx, ety);
        if ((ek == MLIR_TYPE_KIND_INTEGER || ek == MLIR_TYPE_KIND_INDEX) &&
            native_lane_in_bounds(ctx, vty, lane)) {
            if (out) *out = 0;
            return true;
        }
        return false;
    }
    if (!MLIR_OpHasTrait(ctx, def, MLIR_T_CONSTANT)) return false;
    MLIR_AttributeHandle a = MLIR_GetOpAttr(ctx, def, MLIR_KEY_VALUE);
    if (a == MLIR_INVALID_HANDLE) return false;
    const IR_Attr *r = ATTR_REC(ctx, a, "MLIR_MatchConstantInt");
    if (r->kind == AK_INTEGER || r->kind == AK_BOOL) {
        if (lane > 0) return false;
        if (out) *out = (int64_t)r->bits;
        return true;
    }
    if (r->kind == AK_SPLAT) {
        MLIR_TypeHandle sty = TYPE_H(ctx, r->type);
        if (!native_lane_in_bounds(ctx, sty, lane)) return false;
        const IR_Attr *sr = &ATTRS(ctx)[(uint32_t)r->bits];
        if (sr->kind != AK_INTEGER && sr->kind != AK_BOOL) return false;
        if (out) *out = (int64_t)sr->bits;
        return true;
    }
    return false;
}

bool MLIR_IsConstantZero(MLIR_Context *ctx, MLIR_ValueHandle v, int64_t lane) {
    if (v == MLIR_INVALID_HANDLE) return false;
    if (MLIR_GetValueKind(ctx, v) == MLIR_VALUE_OP_RESULT) {
        MLIR_OpHandle def = MLIR_GetValueDefiningOp(ctx, v);
        if (def != MLIR_INVALID_HANDLE) {
            MLIR_OpType t = MLIR_GetOpType(ctx, def);
            if (t == OP_TYPE_LLVM_MLIR_ZERO || t == OP_TYPE_LLVM_MLIR_NULL)
                return native_lane_in_bounds(ctx, MLIR_GetValueType(ctx, v), lane);
        }
    }
    int64_t iv = 0;
    if (MLIR_MatchConstantInt(ctx, v, lane, &iv)) return iv == 0;
    double dv = 0.0;
    if (MLIR_MatchConstantFloat(ctx, v, lane, &dv)) return dv == 0.0;
    return false;
}

// Value creation
MLIR_ValueHandle MLIR_CreateValueBlockArg(MLIR_Context *ctx, string register_name, uint32_t result_index, MLIR_TypeHandle type, MLIR_LocationHandle location) {
    API_CHECK(ctx != NULL, "MLIR_CreateValueBlockArg", "context is NULL");
    uint32_t v = value_create(ctx, 0, result_index, TYPE_IDX(ctx, type, "MLIR_CreateValueBlockArg"),
                              location, register_name);
    VALS(ctx)[v].index |= VAL_BLOCK_ARG;
    return VAL_H(ctx, v);
}

// Block and Region creation
MLIR_BlockHandle MLIR_CreateBlock(MLIR_Context *ctx) {
    API_CHECK(ctx != NULL, "MLIR_CreateBlock", "context is NULL");
    uint32_t b = vec_push(&ctx->blocks, 1);
    IR_Block *blk = &BLOCKS(ctx)[b];
    memset(blk, 0, sizeof(IR_Block));
    if (ctx->arena) {
        blk->ops = (uint32_t *)arena_alloc(ctx->arena, 16 * sizeof(uint32_t));
        blk->cap_ops = 16;
    }
    return BLOCK_H(ctx, b);
}

MLIR_RegionHandle MLIR_CreateRegion(MLIR_Context *ctx) {
    API_CHECK(ctx != NULL, "MLIR_CreateRegion", "context is NULL");
    uint32_t r = vec_push(&ctx->regions, 1);
    memset(&REGIONS(ctx)[r], 0, sizeof(IR_Region));
    return REGION_H(ctx, r);
}

// Replaces the value if the op has `key`; otherwise appends the pair, in
// place when the op's attributes are already the last words of `extra` (the
// common case: attributes set right after creation), else moving them to
// the end of the vector (the old words are left unused).
void MLIR_SetOpAttr(MLIR_Context *ctx, MLIR_OpHandle oh, MLIR_AttrKey key, MLIR_AttributeHandle value) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_SetOpAttr");
    CHECK_OP(ctx, o, "MLIR_SetOpAttr");
    API_CHECK(key != MLIR_KEY_INVALID && key < ctx->keys.n_keys && value != MLIR_INVALID_HANDLE,
              "MLIR_SetOpAttr", "bad attribute key or value");
    uint32_t v = ATTR_IDX(ctx, value, "MLIR_SetOpAttr");
    uint32_t n = OPS(ctx)[o].n_attributes;
    uint32_t old = OPS(ctx)[o].first_attr;
    for (uint32_t i = 0; i < n; i++) {
        if (EXTRA(ctx)[old + 2 * i] == key) {
            EXTRA(ctx)[old + 2 * i + 1] = v;
            return;
        }
    }
    API_CHECK(n < 0xff, "MLIR_SetOpAttr", "too many attributes");
    if (old + 2 * n == ctx->extra.n) {
        vec_push(&ctx->extra, 2);
    } else {
        uint32_t first = vec_push(&ctx->extra, 2 * (n + 1));
        uint32_t *p = EXTRA(ctx);
        for (uint32_t i = 0; i < 2 * n; i++) p[first + i] = p[old + i];
        OPS(ctx)[o].first_attr = old = first;
    }
    EXTRA(ctx)[old + 2 * n] = key;
    EXTRA(ctx)[old + 2 * n + 1] = v;
    OPS(ctx)[o].n_attributes = (uint8_t)(n + 1);
}

// Attribute accessors: one read of the attribute's record each (dense
// arrays and dictionaries also read their words).

// Dense arrays report MLIR_ATTR_KIND_STRING, as before they were typed
// arrays: their printed form is their string.
MLIR_AttrKind MLIR_GetAttributeKind(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    API_CHECK(ah != MLIR_INVALID_HANDLE, "MLIR_GetAttributeKind", "attribute handle is NULL");
    switch (ATTR_REC(ctx, ah, "MLIR_GetAttributeKind")->kind) {
        case AK_INTEGER: return MLIR_ATTR_KIND_INTEGER;
        case AK_FLOAT:   return MLIR_ATTR_KIND_FLOAT;
        case AK_STRING:  return MLIR_ATTR_KIND_STRING;
        case AK_BOOL:    return MLIR_ATTR_KIND_BOOL;
        case AK_ARRAY:   return MLIR_ATTR_KIND_ARRAY;
        case AK_DICT:    return MLIR_ATTR_KIND_DICT;
        case AK_TYPE:    return MLIR_ATTR_KIND_TYPE;
        case AK_DENSE_I32:
        case AK_DENSE_I64: return MLIR_ATTR_KIND_STRING;
        default:         return MLIR_ATTR_KIND_OTHER;
    }
}

// The printed form of a dense array (`array<i32: 1, 2>`), made on first use
// and kept in the persistent arena.
static string dense_text(MLIR_Context *ctx, uint32_t ai) {
    if (ATTRS(ctx)[ai].text) return ATEXT(ctx)[ATTRS(ctx)[ai].text];
    IR_Attr r = ATTRS(ctx)[ai];
    Arena *arena = ctx->persistent;
    strbuf s = strbuf_make();
    strbuf_append(arena, &s, r.kind == AK_DENSE_I32 ? str_lit("array<i32") : str_lit("array<i64"));
    for (uint32_t i = 0; i < r.n; i++) {
        strbuf_append(arena, &s, i == 0 ? str_lit(": ") : str_lit(", "));
        int64_t v = r.kind == AK_DENSE_I32
            ? (int64_t)(int32_t)AX(ctx)[r.first + i]
            : (int64_t)(((uint64_t)AX(ctx)[r.first + 2 * i + 1] << 32) | AX(ctx)[r.first + 2 * i]);
        strbuf_append(arena, &s, format(arena, str_lit("{}"), v));
    }
    strbuf_append(arena, &s, str_lit(">"));
    uint32_t t = vec_push(&ctx->atext, 1);
    ATEXT(ctx)[t] = strbuf_to_string(s);
    ATTRS(ctx)[ai].text = t;
    return ATEXT(ctx)[t];
}

// The text of a string attribute (symbol refs `@name`), or the printed form
// of a dense array; "" for every other kind.
string MLIR_GetAttributeAsString(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    if (ah == MLIR_INVALID_HANDLE) return str_lit("");
    uint32_t ai = ATTR_IDX(ctx, ah, "MLIR_GetAttributeAsString");
    const IR_Attr *r = &ATTRS(ctx)[ai];
    if (r->kind == AK_STRING) {
        string s = { (char *)(uintptr_t)r->bits, r->n };
        return s;
    }
    if (r->kind == AK_DENSE_I32 || r->kind == AK_DENSE_I64) return dense_text(ctx, ai);
    return str_lit("");
}

int64_t MLIR_GetAttributeInteger(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    const IR_Attr *r = ATTR_REC(ctx, ah, "MLIR_GetAttributeInteger");
    CHECK_ATTR_KIND(ah, r->kind == AK_INTEGER || r->kind == AK_BOOL, "MLIR_GetAttributeInteger", "not an integer or bool attribute");
    return (int64_t)r->bits;
}

string MLIR_GetAttributeString(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    uint32_t ai = ATTR_IDX(ctx, ah, "MLIR_GetAttributeString");
    const IR_Attr *r = &ATTRS(ctx)[ai];
    CHECK_ATTR_KIND(ah, r->kind == AK_STRING || r->kind == AK_DENSE_I32 || r->kind == AK_DENSE_I64,
                    "MLIR_GetAttributeString", "not a string attribute");
    if (r->kind != AK_STRING) return dense_text(ctx, ai);
    string s = { (char *)(uintptr_t)r->bits, r->n };
    return s;
}

double MLIR_GetAttributeFloat(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    const IR_Attr *r = ATTR_REC(ctx, ah, "MLIR_GetAttributeFloat");
    CHECK_ATTR_KIND(ah, r->kind == AK_FLOAT, "MLIR_GetAttributeFloat", "not a float attribute");
    double d;
    memcpy(&d, &r->bits, sizeof d);
    return d;
}

MLIR_TypeHandle MLIR_GetAttributeType(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    API_CHECK(ah != MLIR_INVALID_HANDLE, "MLIR_GetAttributeType", "attribute handle is NULL");
    const IR_Attr *r = ATTR_REC(ctx, ah, "MLIR_GetAttributeType");
    if (r->kind != AK_INTEGER && r->kind != AK_FLOAT) return MLIR_INVALID_HANDLE;
    return TYPE_H(ctx, r->type);
}

bool MLIR_GetAttributeBool(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    const IR_Attr *r = ATTR_REC(ctx, ah, "MLIR_GetAttributeBool");
    CHECK_ATTR_KIND(ah, r->kind == AK_BOOL, "MLIR_GetAttributeBool", "not a bool attribute");
    return r->bits != 0;
}

size_t MLIR_GetAttributeArraySize(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    const IR_Attr *r = ATTR_REC(ctx, ah, "MLIR_GetAttributeArraySize");
    CHECK_ATTR_KIND(ah, r->kind == AK_ARRAY, "MLIR_GetAttributeArraySize", "not an array attribute");
    return r->n;
}

MLIR_AttributeHandle MLIR_GetAttributeArrayElement(MLIR_Context *ctx, MLIR_AttributeHandle ah, size_t idx) {
    const IR_Attr *r = ATTR_REC(ctx, ah, "MLIR_GetAttributeArrayElement");
    CHECK_ATTR_KIND(ah, r->kind == AK_ARRAY, "MLIR_GetAttributeArrayElement", "not an array attribute");
    CHECK_INDEX(idx, r->n, "MLIR_GetAttributeArrayElement");
    return ATTR_H(ctx, AX(ctx)[r->first + idx]);
}

size_t MLIR_GetAttributeDictSize(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    const IR_Attr *r = ATTR_REC(ctx, ah, "MLIR_GetAttributeDictSize");
    CHECK_ATTR_KIND(ah, r->kind == AK_DICT, "MLIR_GetAttributeDictSize", "not a dictionary attribute");
    return r->n;
}

MLIR_AttrKey MLIR_GetAttributeDictKey(MLIR_Context *ctx, MLIR_AttributeHandle ah, size_t idx) {
    const IR_Attr *r = ATTR_REC(ctx, ah, "MLIR_GetAttributeDictKey");
    CHECK_ATTR_KIND(ah, r->kind == AK_DICT, "MLIR_GetAttributeDictKey", "not a dictionary attribute");
    CHECK_INDEX(idx, r->n, "MLIR_GetAttributeDictKey");
    if (r->kind != AK_DICT || idx >= r->n) return MLIR_KEY_INVALID;
    return AX(ctx)[r->first + 2 * idx];
}

MLIR_AttributeHandle MLIR_GetAttributeDictElement(MLIR_Context *ctx, MLIR_AttributeHandle ah, size_t idx) {
    const IR_Attr *r = ATTR_REC(ctx, ah, "MLIR_GetAttributeDictElement");
    CHECK_ATTR_KIND(ah, r->kind == AK_DICT, "MLIR_GetAttributeDictElement", "not a dictionary attribute");
    CHECK_INDEX(idx, r->n, "MLIR_GetAttributeDictElement");
    if (r->kind != AK_DICT || idx >= r->n) return MLIR_INVALID_HANDLE;
    return ATTR_H(ctx, AX(ctx)[r->first + 2 * idx + 1]);
}

MLIR_TypeHandle MLIR_GetAttributeTypeValue(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    API_CHECK(ah != MLIR_INVALID_HANDLE, "MLIR_GetAttributeTypeValue", "attribute handle is NULL");
    const IR_Attr *r = ATTR_REC(ctx, ah, "MLIR_GetAttributeTypeValue");
    return r->kind == AK_TYPE ? TYPE_H(ctx, r->type) : MLIR_INVALID_HANDLE;
}

size_t MLIR_GetAttributeDenseSize(MLIR_Context *ctx, MLIR_AttributeHandle ah) {
    const IR_Attr *r = ATTR_REC(ctx, ah, "MLIR_GetAttributeDenseSize");
    return (r->kind == AK_DENSE_I32 || r->kind == AK_DENSE_I64) ? r->n : 0;
}

int64_t MLIR_GetAttributeDenseElement(MLIR_Context *ctx, MLIR_AttributeHandle ah, size_t idx) {
    const IR_Attr *r = ATTR_REC(ctx, ah, "MLIR_GetAttributeDenseElement");
    CHECK_INDEX(idx, (r->kind == AK_DENSE_I32 || r->kind == AK_DENSE_I64) ? r->n : 0,
                "MLIR_GetAttributeDenseElement");
    if (r->kind == AK_DENSE_I32) return (int64_t)(int32_t)AX(ctx)[r->first + idx];
    return (int64_t)(((uint64_t)AX(ctx)[r->first + 2 * idx + 1] << 32) | AX(ctx)[r->first + 2 * idx]);
}

size_t MLIR_GetAttributeDenseI32(MLIR_Context *ctx, MLIR_AttributeHandle ah,
                                 int32_t *out, size_t cap) {
    const IR_Attr *r = ATTR_REC(ctx, ah, "MLIR_GetAttributeDenseI32");
    if (r->kind == AK_DENSE_I32) {
        size_t n = r->n;
        size_t copy = n < cap ? n : cap;
        if (copy && out) memcpy(out, AX(ctx) + r->first, copy * sizeof(int32_t));
        return n;
    }
    if (r->kind == AK_DENSE_I64) {
        size_t n = r->n;
        size_t copy = n < cap ? n : cap;
        if (copy && out) {
            const uint32_t *ax = AX(ctx) + r->first;
            for (size_t i = 0; i < copy; i++) out[i] = (int32_t)ax[2 * i];
        }
        return n;
    }
    return 0;
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
    return TYPE_H(ctx, VALS(ctx)[v].type);
}

string MLIR_GetValueRegisterName(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_GetValueRegisterName");
    if (!v || !(VALS(ctx)[v].index & VAL_HAS_NAME)) return str_lit("");
    uint32_t n = u32map_get(&ctx->vname_map, v);
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
    return (p->index & VAL_BLOCK_ARG) ? MLIR_INVALID_HANDLE : OP_H(ctx, p->def);
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

// Locations: records in the `locs` vector (see "Locations" above).

static MLIR_LocationHandle loc_create(MLIR_Context *ctx, uint32_t kind, string str, int line, int col) {
    API_CHECK(ctx != NULL, "MLIR_CreateLocation", "context is NULL");
    uint32_t si = str.size ? attr_key_table_intern(&ctx->loc_strings, str) : 0;
    uint32_t l = vec_push(&ctx->locs, 1);
    IR_Loc *p = &LOCS(ctx)[l];
    p->kind = kind;
    p->str = si;
    p->line = line;
    p->col = col;
    p->text = 0;
    return LOC_H(ctx, l);
}

static string loc_str(MLIR_Context *ctx, const IR_Loc *l) {
    return attr_key_table_name(&ctx->loc_strings, l->str);
}

#define LOC_REC(ctx, h, fn) (&LOCS(ctx)[LOC_IDX((ctx), (h), (fn))])

MLIR_LocationKind MLIR_GetLocationKind(MLIR_Context *ctx, MLIR_LocationHandle lh) {
    return (MLIR_LocationKind)LOC_REC(ctx, lh, "MLIR_GetLocationKind")->kind;
}

// The parser's text of an unknown location; the `loc(...)` form of the
// others, made on the first call and kept in the context.
string MLIR_GetLocationOriginalText(MLIR_Context *ctx, MLIR_LocationHandle lh) {
    uint32_t li = LOC_IDX(ctx, lh, "MLIR_GetLocationOriginalText");
    IR_Loc l = LOCS(ctx)[li];
    if (l.kind != MLIR_LOC_FILE && l.kind != MLIR_LOC_NAME && l.kind != MLIR_LOC_REF)
        return loc_str(ctx, &l);
    if (l.text) return attr_key_table_name(&ctx->loc_strings, l.text);
    string text;
    if (l.kind == MLIR_LOC_FILE)
        text = format(ctx->persistent, str_lit("loc({}:{}:{})"), loc_str(ctx, &l),
                      (int64_t)l.line, (int64_t)l.col);
    else if (l.kind == MLIR_LOC_NAME)
        text = format(ctx->persistent, str_lit("loc(\"{}\")"), loc_str(ctx, &l));
    else
        text = format(ctx->persistent, str_lit("loc(#loc{})"), (int64_t)l.line);
    uint32_t t = attr_key_table_intern(&ctx->loc_strings, text);
    LOCS(ctx)[li].text = t;
    return attr_key_table_name(&ctx->loc_strings, t);
}

string MLIR_GetLocationFileFilename(MLIR_Context *ctx, MLIR_LocationHandle lh) {
    const IR_Loc *l = LOC_REC(ctx, lh, "MLIR_GetLocationFileFilename");
    return (l->kind == MLIR_LOC_FILE || l->kind == MLIR_LOC_NAME) ? loc_str(ctx, l) : str_lit("");
}

int MLIR_GetLocationFileLine(MLIR_Context *ctx, MLIR_LocationHandle lh) {
    const IR_Loc *l = LOC_REC(ctx, lh, "MLIR_GetLocationFileLine");
    return l->kind == MLIR_LOC_FILE ? l->line : 0;
}

int MLIR_GetLocationFileColumn(MLIR_Context *ctx, MLIR_LocationHandle lh) {
    const IR_Loc *l = LOC_REC(ctx, lh, "MLIR_GetLocationFileColumn");
    return l->kind == MLIR_LOC_FILE ? l->col : 0;
}

string MLIR_GetLocationName(MLIR_Context *ctx, MLIR_LocationHandle lh) {
    const IR_Loc *l = LOC_REC(ctx, lh, "MLIR_GetLocationName");
    return (l->kind == MLIR_LOC_FILE || l->kind == MLIR_LOC_NAME) ? loc_str(ctx, l) : str_lit("");
}

// Without text, the shared unknown location (element 1): creating one per op
// costs nothing.
MLIR_LocationHandle MLIR_CreateLocationUnknown(MLIR_Context *ctx, string original_text) {
    if (original_text.size == 0) return LOC_H(ctx, 1);
    return loc_create(ctx, MLIR_LOC_UNKNOWN, original_text, 0, 0);
}

MLIR_LocationHandle MLIR_CreateLocationFile(MLIR_Context *ctx, string filename, int line, int column) {
    return loc_create(ctx, MLIR_LOC_FILE, filename, line, column);
}

MLIR_LocationHandle MLIR_CreateLocationName(MLIR_Context *ctx, string name) {
    return loc_create(ctx, MLIR_LOC_NAME, name, 0, 0);
}

MLIR_LocationHandle MLIR_CreateLocationRef(MLIR_Context *ctx, int ref_id) {
    return loc_create(ctx, MLIR_LOC_REF, (string){0}, ref_id, 0);
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
    if (MLIR_NAT_LIKELY(keep)) {
        uint32_t tail = head;
        for (uint32_t u = head; u; u = rec[u].next) {
            rec[u].value = to;
            tail = u;
        }
        // Splice the chain in front of the new value's uses.
        uint32_t to_head = vals[to].first_use;
        rec[tail].next = to_head;
        if (to_head) rec[to_head].prev = tail;
        vals[to].first_use = head;
        vals[to].n_uses += vals[from].n_uses;
    } else {
        for (uint32_t u = head; u; ) {
            uint32_t next = rec[u].next;
            rec[u].value = to;
            rec[u].next = 0;
            rec[u].prev = 0;
            u = next;
        }
    }
    vals[from].first_use = 0;
    vals[from].n_uses = 0;
}

static MLIR_NAT_NOINLINE void detach_from_block_slow(MLIR_Context *ctx, IR_Block *blk,
                                                     uint32_t pos, uint32_t o) {
    if (pos < blk->n_ops && blk->ops[pos] == o) {
        memmove(blk->ops + pos, blk->ops + pos + 1, (blk->n_ops - pos - 1) * sizeof(uint32_t));
        blk->n_ops--;
        for (uint32_t i = pos; i < blk->n_ops; i++) OPS(ctx)[blk->ops[i]].block_pos = i;
    } else {
        u32_remove(blk->ops, &blk->n_ops, o);
        for (uint32_t i = 0; i < blk->n_ops; i++) OPS(ctx)[blk->ops[i]].block_pos = i;
    }
}

// Removes op `o` from its parent block's op list (O(1) when last, O(n) otherwise).
static inline void detach_from_block(MLIR_Context *ctx, uint32_t o) {
    IR_Op *op = &OPS(ctx)[o];
    uint32_t b = op->parent_block;
    if (MLIR_NAT_LIKELY(b != 0)) {
        IR_Block *blk = &BLOCKS(ctx)[b];
        uint32_t pos = op->block_pos;
#ifdef MLIR_API_CHECKS
        if (MLIR_NAT_LIKELY(pos + 1 == blk->n_ops && blk->ops[pos] == o)) {
#else
        if (MLIR_NAT_LIKELY(pos + 1 == blk->n_ops)) {
#endif
            blk->n_ops = pos;
        } else {
            detach_from_block_slow(ctx, blk, pos, o);
        }
    }
    op->parent_block = 0;
    op->block_pos = 0;
}

static void drop_op_uses(MLIR_Context *ctx, uint32_t o);

static MLIR_NAT_NOINLINE void drop_op_extra_slow(MLIR_Context *ctx, uint32_t o) {
    const IR_OpExtra *ox = op_extra_get(ctx, o);
    if (!ox) return;
    IR_Op *op = &OPS(ctx)[o];
    if ((op->flags & OPF_TRACKED) && ox->n_succ_operands > 0) {
        uint32_t first = op->first_operand + op->n_operands;
        uint32_t n_succ = ox->n_succ_operands;
        IR_Operand *rec = OPNDS(ctx);
        IR_Value *vals = VALS(ctx);
        for (uint32_t i = 0; i < n_succ; i++) {
            uint32_t u = first + i;
            uint32_t v = rec[u].value;
            if (v != 0 && (vals[v].index & VAL_TRACKED)) {
                uint32_t prev = rec[u].prev;
                uint32_t next = rec[u].next;
                if (prev) rec[prev].next = next;
                else      vals[v].first_use = next;
                if (next) rec[next].prev = prev;
                rec[u].next = 0;
                rec[u].prev = 0;
                if (vals[v].n_uses) vals[v].n_uses--;
            }
            rec[u].value = 0;
        }
    }
    for (uint32_t s = 0; s < ox->n_successors; s++) {
        uint32_t pe_idx = op_succ_pair(ctx, o, s)[2];
        if (pe_idx) pred_unlink(ctx, pe_idx);
    }
    uint32_t nr = ox->n_regions;
    uint32_t fe = ox->first_extra;
    for (uint32_t r = 0; r < nr; r++) {
        IR_Region *reg = &REGIONS(ctx)[EXTRA(ctx)[fe + r]];
        for (uint32_t bi = 0; bi < reg->n_blocks; bi++) {
            IR_Block *blk = &BLOCKS(ctx)[reg->blocks[bi]];
            for (uint32_t k = 0; k < blk->n_ops; k++) drop_op_uses(ctx, blk->ops[k]);
        }
    }
}

static inline void unlink_one_operand(IR_Operand *rec, IR_Value *vals, uint32_t u) {
    uint32_t v = rec[u].value;
    if (MLIR_NAT_LIKELY(v != 0 && (vals[v].index & VAL_TRACKED))) {
        uint32_t prev = rec[u].prev;
        uint32_t next = rec[u].next;
        if (prev) rec[prev].next = next;
        else      vals[v].first_use = next;
        if (next) rec[next].prev = prev;
        vals[v].n_uses--;
    }
#ifdef MLIR_API_CHECKS
    rec[u].next = 0;
    rec[u].prev = 0;
    rec[u].value = 0;
#endif
}

static MLIR_NAT_NOINLINE void drop_op_uses_many_slow(MLIR_Context *ctx, uint32_t first, uint32_t n_reg) {
    IR_Operand *rec = OPNDS(ctx);
    IR_Value *vals = VALS(ctx);
    for (uint32_t i = 0; i < n_reg; i++) unlink_one_operand(rec, vals, first + i);
}

// Unlinks every operand use and predecessor edge of op `o` and of the ops
// nested in its regions, as upstream's Operation::erase does when it destroys
// them. Under MLIR_API_CHECKS the ops are also marked erased.
static inline void drop_op_uses_inline(MLIR_Context *ctx, uint32_t o) {
    IR_Op *op = &OPS(ctx)[o];
    if (op->flags & OPF_TRACKED) {
        uint32_t first = op->first_operand;
        uint32_t n_reg = op->n_operands;
        if (MLIR_NAT_LIKELY(n_reg <= 2)) {
            if (n_reg > 0) {
                IR_Operand *rec = OPNDS(ctx);
                IR_Value *vals = VALS(ctx);
                unlink_one_operand(rec, vals, first);
                if (n_reg == 2) unlink_one_operand(rec, vals, first + 1);
            }
        } else {
            drop_op_uses_many_slow(ctx, first, n_reg);
        }
    }
    if (MLIR_NAT_UNLIKELY(op->flags & OPF_HAS_EXTRA))
        drop_op_extra_slow(ctx, o);
#ifdef MLIR_API_CHECKS
    OPS(ctx)[o].op_type = OP_TYPE_ERASED_SENTINEL;
#endif
}

static void drop_op_uses(MLIR_Context *ctx, uint32_t o) {
    drop_op_uses_inline(ctx, o);
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
        if (!op_is_inside(ctx, OPNDS(ctx)[u].owner & OPND_OWNER_MASK, root)) return true;
    return false;
}

// True if a value defined in op `o` or nested in it is used outside `root`.
static bool op_used_outside(MLIR_Context *ctx, uint32_t o, uint32_t root) {
    IR_Op *op = &OPS(ctx)[o];
    for (uint32_t i = 0; i < op->n_results; i++)
        if (value_used_outside(ctx, op->first_result + i, root)) return true;
    const IR_OpExtra *ox = op_extra_get(ctx, o);
    if (ox) {
        for (uint32_t r = 0; r < ox->n_regions; r++) {
            IR_Region *reg = &REGIONS(ctx)[EXTRA(ctx)[ox->first_extra + r]];
            for (uint32_t bi = 0; bi < reg->n_blocks; bi++) {
                IR_Block *b = &BLOCKS(ctx)[reg->blocks[bi]];
                for (uint32_t i = 0; i < b->n_args; i++)
                    if (value_used_outside(ctx, b->args[i], root)) return true;
                for (uint32_t k = 0; k < b->n_ops; k++)
                    if (op_used_outside(ctx, b->ops[k], root)) return true;
            }
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
    drop_op_uses_inline(ctx, o);  // also marks the op erased
#else
    // Without def-use tracking and regions/successors there is nothing to drop.
    if (OPS(ctx)[o].flags & (OPF_TRACKED | OPF_HAS_EXTRA))
        drop_op_uses_inline(ctx, o);
#endif
}

void MLIR_EraseOps(MLIR_Context *ctx, const MLIR_OpHandle *ops, size_t n) {
    if (n == 0) return;
    if (n == 1) { MLIR_EraseOp(ctx, ops[0]); return; }
#ifdef MLIR_API_CHECKS
    for (size_t i = 0; i < n; i++) MLIR_EraseOp(ctx, ops[i]);
#else
    uint32_t blks_stk[64];
    uint32_t *blks = blks_stk;
    size_t n_blks = 0, cap_blks = 64;
    IR_Op *all_ops = OPS(ctx);
    IR_Block *all_blks = BLOCKS(ctx);
    for (size_t i = 0; i < n; i++) {
        uint32_t o = OP_IDX(ctx, ops[i], "MLIR_EraseOps");
        if (!o) continue;
        uint32_t b = all_ops[o].parent_block;
        if (b != 0) {
            IR_Block *blk = &all_blks[b];
            uint32_t pos = all_ops[o].block_pos;
            if (pos < blk->n_ops && blk->ops[pos] == o) {
                blk->ops[pos] = 0;
                if (n_blks == 0 || blks[n_blks - 1] != b) {
                    if (n_blks == cap_blks) {
                        size_t ncap = cap_blks * 2;
                        uint32_t *nb = (uint32_t *)malloc(ncap * sizeof(uint32_t));
                        memcpy(nb, blks, n_blks * sizeof(uint32_t));
                        if (blks != blks_stk) free(blks);
                        blks = nb; cap_blks = ncap;
                    }
                    blks[n_blks++] = b;
                }
            } else {
                detach_from_block(ctx, o);
            }
            all_ops[o].parent_block = 0;
            all_ops[o].block_pos = 0;
        }
        if (all_ops[o].flags & (OPF_TRACKED | OPF_HAS_EXTRA))
            drop_op_uses_inline(ctx, o);
    }
    for (size_t bi = 0; bi < n_blks; bi++) {
        IR_Block *blk = &all_blks[blks[bi]];
        uint32_t w = 0, no = blk->n_ops;
        uint32_t *bops = blk->ops;
        for (uint32_t r = 0; r < no; r++) {
            uint32_t op = bops[r];
            if (op != 0) {
                if (w != r) {
                    bops[w] = op;
                    all_ops[op].block_pos = w;
                }
                w++;
            }
        }
        blk->n_ops = w;
    }
    if (blks != blks_stk) free(blks);
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
    const IR_OpExtra *ox = o ? op_extra_get(ctx, o) : NULL;
    if (!ox || idx >= ox->n_regions) return MLIR_INVALID_HANDLE;
    uint32_t slot = ox->first_extra + (uint32_t)idx;
    // Replace it with a fresh empty region; the taken one is detached until
    // it is passed to MLIR_CreateOp.
    MLIR_RegionHandle fresh_h = MLIR_CreateRegion(ctx);
    uint32_t fresh = REGION_IDX(ctx, fresh_h, "MLIR_TakeOpRegion");
    REGIONS(ctx)[fresh].parent_op = o;
    uint32_t taken = EXTRA(ctx)[slot];
    EXTRA(ctx)[slot] = fresh;
    if (taken) REGIONS(ctx)[taken].parent_op = 0;
    return REGION_H(ctx, taken);
}

MLIR_BlockHandle MLIR_GetOpParentBlock(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpParentBlock");
    CHECK_NOT_ERASED(ctx, o, "MLIR_GetOpParentBlock");
    return BLOCK_H(ctx, OPS(ctx)[o].parent_block);
}

MLIR_RegionHandle MLIR_GetBlockParentRegion(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockParentRegion");
    return REGION_H(ctx, BLOCKS(ctx)[b].parent_region);
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
        uint32_t pos = BLOCKS(ctx)[b].region_pos;
        if (pos < reg->n_blocks && reg->blocks[pos] == b) {
            if (pos + 1 < reg->n_blocks)
                memmove(reg->blocks + pos, reg->blocks + pos + 1,
                        (reg->n_blocks - pos - 1) * sizeof(uint32_t));
            reg->n_blocks--;
            for (uint32_t i = pos; i < reg->n_blocks; i++)
                BLOCKS(ctx)[reg->blocks[i]].region_pos = i;
        } else {
            u32_remove(reg->blocks, &reg->n_blocks, b);
            for (uint32_t i = 0; i < reg->n_blocks; i++)
                BLOCKS(ctx)[reg->blocks[i]].region_pos = i;
        }
    }
    BLOCKS(ctx)[b].parent_region = 0;
    BLOCKS(ctx)[b].region_pos = 0;
}

void MLIR_MoveBlockToRegionEnd(MLIR_Context *ctx, MLIR_BlockHandle bh, MLIR_RegionHandle dest) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_MoveBlockToRegionEnd");
    uint32_t r = REGION_IDX(ctx, dest, "MLIR_MoveBlockToRegionEnd");
    if (b == 0 || r == 0) return;
    detach_from_region(ctx, b);
    MLIR_AppendRegionBlock(ctx, dest, bh);
}

void MLIR_MoveBlocksToRegionEnd(MLIR_Context *ctx, const MLIR_BlockHandle *bhs,
                                size_t n_blocks, MLIR_RegionHandle dest) {
    if (n_blocks == 0) return;
    if (n_blocks == 1) {
        MLIR_MoveBlockToRegionEnd(ctx, bhs[0], dest);
        return;
    }
    uint32_t dst_r = REGION_IDX(ctx, dest, "MLIR_MoveBlocksToRegionEnd");
    if (dst_r == 0) return;
    // Fast path: all blocks currently belong to the same source region (and not dst_r).
    uint32_t first_b = BLOCK_IDX(ctx, bhs[0], "MLIR_MoveBlocksToRegionEnd");
    uint32_t src_r = first_b ? BLOCKS(ctx)[first_b].parent_region : 0;
    bool same_src = (src_r != 0 && src_r != dst_r);
    if (same_src) {
        for (size_t i = 1; i < n_blocks; i++) {
            uint32_t b = BLOCK_IDX(ctx, bhs[i], "MLIR_MoveBlocksToRegionEnd");
            if (b == 0 || BLOCKS(ctx)[b].parent_region != src_r) {
                same_src = false;
                break;
            }
        }
    }
    if (!same_src) {
        for (size_t i = 0; i < n_blocks; i++)
            MLIR_MoveBlockToRegionEnd(ctx, bhs[i], dest);
        return;
    }
    IR_Block *blks = BLOCKS(ctx);
    for (size_t i = 0; i < n_blocks; i++) {
        uint32_t b = BLOCK_IDX(ctx, bhs[i], "MLIR_MoveBlocksToRegionEnd");
        blks[b].parent_region = 0;
    }
    IR_Region *sreg = &REGIONS(ctx)[src_r];
    uint32_t w = 0;
    for (uint32_t i = 0; i < sreg->n_blocks; i++) {
        uint32_t x = sreg->blocks[i];
        if (blks[x].parent_region == src_r) {
            sreg->blocks[w] = x;
            blks[x].region_pos = w;
            w++;
        }
    }
    sreg->n_blocks = w;

    IR_Region *dreg = &REGIONS(ctx)[dst_r];
    dreg->blocks = u32_reserve(ctx->arena, dreg->blocks, dreg->n_blocks,
                               &dreg->cap_blocks, (size_t)dreg->n_blocks + n_blocks);
    for (size_t i = 0; i < n_blocks; i++) {
        uint32_t b = BLOCK_IDX(ctx, bhs[i], "MLIR_MoveBlocksToRegionEnd");
        uint32_t pos = dreg->n_blocks++;
        dreg->blocks[pos] = b;
        blks[b].parent_region = dst_r;
        blks[b].region_pos = pos;
    }
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
    const IR_OpExtra *ox = op_extra_get(ctx, o);
    size_t n_total = (size_t)op->n_operands + (ox ? ox->n_succ_operands : 0);
    // idx out of range: silently no-op (as before).
    if (idx >= n_total) return;
    operand_set(ctx, o, op->first_operand + (uint32_t)idx, v);
}

void MLIR_SetOpSuccessor(MLIR_Context *ctx, MLIR_OpHandle oh, size_t succ_idx, MLIR_BlockHandle block) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_SetOpSuccessor");
    uint32_t b = BLOCK_IDX(ctx, block, "MLIR_SetOpSuccessor");
    CHECK_NOT_ERASED(ctx, o, "MLIR_SetOpSuccessor");
    const IR_OpExtra *ox = o ? op_extra_get(ctx, o) : NULL;
    if (!ox || succ_idx >= ox->n_successors) return;
    uint32_t *triple = op_succ_pair(ctx, o, succ_idx);
    if (triple[0] == b) return;
    uint32_t pe_idx = triple[2];
    if (pe_idx) {
        pred_unlink(ctx, pe_idx);
        PREDS(ctx)[pe_idx].target_block = b;
        if (b) pred_push(ctx, pe_idx);
    }
    triple[0] = b;
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
    IR_OpExtra *ox = op_extra_mut(ctx, o);
    uint32_t old_total = (uint32_t)OPS(ctx)[o].n_operands + ox->n_succ_operands;
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
        rec->owner = opnd_pack_owner(o, base + k);
        rec->next = rec->prev = 0;
    }
    for (uint32_t i = base + old_n; i < old_total; i++) {
        uint32_t dst_off = i - old_n + new_n;
        operand_move(ctx, old_first + i, first + dst_off);
        OPNDS(ctx)[first + dst_off].owner = opnd_pack_owner(o, dst_off);
    }
    GEN_CHECK(ctx, opnds, "op_resize_succ_segment");
    IR_Op *op = &OPS(ctx)[o];
    op->first_operand = first;
    ox = op_extra_mut(ctx, o);
    ox->n_succ_operands = (uint16_t)(ox->n_succ_operands - old_n + new_n);
    op_succ_pair(ctx, o, s)[1] = new_n;
}

void MLIR_SetOpSuccessorOperands(MLIR_Context *ctx, MLIR_OpHandle oh, size_t succ_idx,
                                 const MLIR_ValueHandle *values, size_t n) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_SetOpSuccessorOperands");
    CHECK_NOT_ERASED(ctx, o, "MLIR_SetOpSuccessorOperands");
    const IR_OpExtra *ox = o ? op_extra_get(ctx, o) : NULL;
    if (!ox || succ_idx >= ox->n_successors) return;
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
    const IR_OpExtra *ox = o ? op_extra_get(ctx, o) : NULL;
    if (!ox || succ_idx >= ox->n_successors) return;
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
            const IR_OpExtra *ox = op_extra_get(ctx, term);
            uint32_t ns = ox ? ox->n_successors : 0;
            for (uint32_t k = 0; k < ns; k++)
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
    for (uint32_t i = at; i < reg->n_blocks; i++)
        BLOCKS(ctx)[reg->blocks[i]].region_pos = i;
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
        at = (BLOCKS(ctx)[a].parent_region == r) ? (BLOCKS(ctx)[a].region_pos + 1) : reg->n_blocks;
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
    uint32_t at = (bf != 0 && BLOCKS(ctx)[bf].parent_region == r)
        ? BLOCKS(ctx)[bf].region_pos : reg->n_blocks;
    region_insert_block(ctx, r, b, at);
}

MLIR_OpHandle MLIR_GetBlockTerminator(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockTerminator");
    IR_Block *blk = &BLOCKS(ctx)[b];
    return blk->n_ops ? OP_H(ctx, blk->ops[blk->n_ops - 1]) : MLIR_INVALID_HANDLE;
}

MLIR_OpHandle MLIR_GetBlockParentOp(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockParentOp");
    uint32_t r = BLOCKS(ctx)[b].parent_region;
    return OP_H(ctx, REGIONS(ctx)[r].parent_op);
}

bool MLIR_IsBlockEntry(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_IsBlockEntry");
    if (!b) return false;
    uint32_t r = BLOCKS(ctx)[b].parent_region;
    return r && REGIONS(ctx)[r].n_blocks > 0 && REGIONS(ctx)[r].blocks[0] == b;
}

// Predecessors: walks the block's O(in-degree) predecessor edge list.
size_t MLIR_GetBlockNumPredecessors(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockNumPredecessors");
    if (!b) return 0;
    uint32_t r = BLOCKS(ctx)[b].parent_region;
    if (!r) return 0;
    const IR_PredEdge *preds = PREDS(ctx);
    const IR_Op *ops = OPS(ctx);
    const IR_Block *blocks = BLOCKS(ctx);
    size_t n = 0;
    for (uint32_t p = blocks[b].first_pred; p; p = preds[p].next) {
        uint32_t pb = ops[preds[p].owner].parent_block;
        if (pb && blocks[pb].parent_region == r) n++;
    }
    return n;
}

MLIR_BlockHandle MLIR_GetBlockPredecessor(MLIR_Context *ctx, MLIR_BlockHandle bh, size_t idx,
                                          size_t *out_succ_idx) {
    if (out_succ_idx) *out_succ_idx = SIZE_MAX;
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockPredecessor");
    if (!b) return MLIR_INVALID_HANDLE;
    uint32_t r = BLOCKS(ctx)[b].parent_region;
    if (!r) return MLIR_INVALID_HANDLE;
    const IR_PredEdge *preds = PREDS(ctx);
    const IR_Op *ops = OPS(ctx);
    const IR_Block *blocks = BLOCKS(ctx);
    size_t n = 0;
    for (uint32_t p = blocks[b].first_pred; p; p = preds[p].next) {
        uint32_t pb = ops[preds[p].owner].parent_block;
        if (pb && blocks[pb].parent_region == r) {
            if (n == idx) {
                if (out_succ_idx) *out_succ_idx = preds[p].succ_idx;
                return BLOCK_H(ctx, pb);
            }
            n++;
        }
    }
    return MLIR_INVALID_HANDLE;
}

MLIR_ValueHandle MLIR_AddBlockArgument(MLIR_Context *ctx, MLIR_BlockHandle bh,
                                       MLIR_TypeHandle type, MLIR_LocationHandle loc) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_AddBlockArgument");
    if (!b) return MLIR_INVALID_HANDLE;
    uint32_t v = value_create(ctx, b, BLOCKS(ctx)[b].n_args,
                              TYPE_IDX(ctx, type, "MLIR_AddBlockArgument"), loc, (string){0});
    VALS(ctx)[v].index |= VAL_BLOCK_ARG;
    IR_Block *blk = &BLOCKS(ctx)[b];
    blk->args = u32_reserve(ctx->arena, blk->args, blk->n_args, &blk->cap_args, (size_t)blk->n_args + 1);
    blk->args[blk->n_args++] = v;
    return VAL_H(ctx, v);
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
    if (p->index & VAL_BLOCK_ARG) return BLOCK_H(ctx, p->def);
    return BLOCK_H(ctx, OPS(ctx)[p->def].parent_block);
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
    uint32_t raw_owner = OPNDS(ctx)[u].owner;
    uint32_t owner = raw_owner & OPND_OWNER_MASK;
    if (out_operand_idx) {
        uint32_t oidx = raw_owner >> OPND_IDX_SHIFT;
        *out_operand_idx = (oidx < OPND_IDX_ESCAPE) ? (size_t)oidx : (size_t)(u - OPS(ctx)[owner].first_operand);
    }
    return OP_H(ctx, owner);
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
    for (uint32_t i = 0; i < sb->n_ops; i++) {
        OPS(ctx)[sb->ops[i]].parent_block = d;
        OPS(ctx)[sb->ops[i]].block_pos = db->n_ops + i;
    }
    db->n_ops = (uint32_t)new_n;
    sb->ops = NULL;
    sb->n_ops = 0;
    sb->cap_ops = 0;
}

// -----------------------------------------------------------------------------
// O(1) Traversal iterators (#352)
// -----------------------------------------------------------------------------

MLIR_OpHandle MLIR_GetBlockFirstOp(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockFirstOp");
    if (!b) return MLIR_INVALID_HANDLE;
    IR_Block *blk = &BLOCKS(ctx)[b];
    return blk->n_ops ? OP_H(ctx, blk->ops[0]) : MLIR_INVALID_HANDLE;
}

MLIR_OpHandle MLIR_GetBlockLastOp(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockLastOp");
    if (!b) return MLIR_INVALID_HANDLE;
    IR_Block *blk = &BLOCKS(ctx)[b];
    return blk->n_ops ? OP_H(ctx, blk->ops[blk->n_ops - 1]) : MLIR_INVALID_HANDLE;
}

MLIR_OpHandle MLIR_GetOpNext(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpNext");
    CHECK_OP(ctx, o, "MLIR_GetOpNext");
    uint32_t b = OPS(ctx)[o].parent_block;
    if (!b) return MLIR_INVALID_HANDLE;
    IR_Block *blk = &BLOCKS(ctx)[b];
    uint32_t next_pos = OPS(ctx)[o].block_pos + 1;
    return next_pos < blk->n_ops ? OP_H(ctx, blk->ops[next_pos]) : MLIR_INVALID_HANDLE;
}

MLIR_OpHandle MLIR_GetOpPrev(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetOpPrev");
    CHECK_OP(ctx, o, "MLIR_GetOpPrev");
    uint32_t b = OPS(ctx)[o].parent_block;
    if (!b) return MLIR_INVALID_HANDLE;
    uint32_t pos = OPS(ctx)[o].block_pos;
    return pos > 0 ? OP_H(ctx, BLOCKS(ctx)[b].ops[pos - 1]) : MLIR_INVALID_HANDLE;
}

MLIR_BlockOpIter MLIR_BlockOpsBegin(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    MLIR_BlockOpIter it;
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_BlockOpsBegin");
    if (!b) {
        it.cur = 0;
        it.end = 0;
        return it;
    }
    IR_Block *blk = &BLOCKS(ctx)[b];
    it.cur = (uintptr_t)blk->ops;
    it.end = blk->n_ops ? (uintptr_t)(blk->ops + blk->n_ops) : it.cur;
    return it;
}

MLIR_OpHandle MLIR_BlockOpIterNext(MLIR_Context *ctx, MLIR_BlockOpIter *it) {
    if (it->cur == it->end) return MLIR_INVALID_HANDLE;
    uint32_t *p = (uint32_t *)it->cur;
    it->cur = (uintptr_t)(p + 1);
    return OP_H(ctx, *p);
}

MLIR_OpHandle MLIR_GetParentOp(MLIR_Context *ctx, MLIR_OpHandle oh) {
    uint32_t o = OP_IDX(ctx, oh, "MLIR_GetParentOp");
    CHECK_OP(ctx, o, "MLIR_GetParentOp");
    uint32_t b = OPS(ctx)[o].parent_block;
    if (!b) return MLIR_INVALID_HANDLE;
    uint32_t r = BLOCKS(ctx)[b].parent_region;
    return r ? OP_H(ctx, REGIONS(ctx)[r].parent_op) : MLIR_INVALID_HANDLE;
}

MLIR_BlockHandle MLIR_GetRegionFirstBlock(MLIR_Context *ctx, MLIR_RegionHandle rh) {
    uint32_t r = REGION_IDX(ctx, rh, "MLIR_GetRegionFirstBlock");
    if (!r) return MLIR_INVALID_HANDLE;
    IR_Region *reg = &REGIONS(ctx)[r];
    return reg->n_blocks ? BLOCK_H(ctx, reg->blocks[0]) : MLIR_INVALID_HANDLE;
}

MLIR_BlockHandle MLIR_GetRegionLastBlock(MLIR_Context *ctx, MLIR_RegionHandle rh) {
    uint32_t r = REGION_IDX(ctx, rh, "MLIR_GetRegionLastBlock");
    if (!r) return MLIR_INVALID_HANDLE;
    IR_Region *reg = &REGIONS(ctx)[r];
    return reg->n_blocks ? BLOCK_H(ctx, reg->blocks[reg->n_blocks - 1]) : MLIR_INVALID_HANDLE;
}

MLIR_BlockHandle MLIR_GetBlockNext(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockNext");
    if (!b) return MLIR_INVALID_HANDLE;
    uint32_t r = BLOCKS(ctx)[b].parent_region;
    if (!r) return MLIR_INVALID_HANDLE;
    IR_Region *reg = &REGIONS(ctx)[r];
    uint32_t next_pos = BLOCKS(ctx)[b].region_pos + 1;
    return next_pos < reg->n_blocks ? BLOCK_H(ctx, reg->blocks[next_pos]) : MLIR_INVALID_HANDLE;
}

MLIR_BlockHandle MLIR_GetBlockPrev(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockPrev");
    if (!b) return MLIR_INVALID_HANDLE;
    uint32_t r = BLOCKS(ctx)[b].parent_region;
    if (!r) return MLIR_INVALID_HANDLE;
    uint32_t pos = BLOCKS(ctx)[b].region_pos;
    return pos > 0 ? BLOCK_H(ctx, REGIONS(ctx)[r].blocks[pos - 1]) : MLIR_INVALID_HANDLE;
}

MLIR_UseHandle MLIR_GetValueFirstUse(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_GetValueFirstUse");
    if (!v) return MLIR_INVALID_HANDLE;
    return (MLIR_UseHandle)VALS(ctx)[v].first_use;
}

MLIR_UseHandle MLIR_GetUseNext(MLIR_Context *ctx, MLIR_UseHandle use) {
    API_CHECK(ctx != NULL, "MLIR_GetUseNext", "context is NULL");
    uint32_t u = (uint32_t)use;
    if (!u) return MLIR_INVALID_HANDLE;
    API_CHECK(u < ctx->opnds.n, "MLIR_GetUseNext", "invalid use handle");
    return (MLIR_UseHandle)OPNDS(ctx)[u].next;
}

MLIR_OpHandle MLIR_GetUseOwner(MLIR_Context *ctx, MLIR_UseHandle use) {
    API_CHECK(ctx != NULL, "MLIR_GetUseOwner", "context is NULL");
    uint32_t u = (uint32_t)use;
    API_CHECK(u > 0 && u < ctx->opnds.n, "MLIR_GetUseOwner", "invalid use handle");
    return OP_H(ctx, OPNDS(ctx)[u].owner & OPND_OWNER_MASK);
}

size_t MLIR_GetUseOperandIndex(MLIR_Context *ctx, MLIR_UseHandle use) {
    API_CHECK(ctx != NULL, "MLIR_GetUseOperandIndex", "context is NULL");
    uint32_t u = (uint32_t)use;
    API_CHECK(u > 0 && u < ctx->opnds.n, "MLIR_GetUseOperandIndex", "invalid use handle");
    uint32_t raw = OPNDS(ctx)[u].owner;
    uint32_t idx = raw >> OPND_IDX_SHIFT;
    if (MLIR_NAT_LIKELY(idx < OPND_IDX_ESCAPE)) return (size_t)idx;
    return (size_t)(u - OPS(ctx)[raw & OPND_OWNER_MASK].first_operand);
}

static inline int32_t nat_cfg_intersect(const int32_t *idom, int32_t b1, int32_t b2) {
    while (b1 != b2) {
        while (b1 > b2) b1 = idom[b1];
        while (b2 > b1) b2 = idom[b2];
    }
    return b1;
}

MLIR_CFGInfo *MLIR_ComputeCFGInfo(MLIR_Context *ctx, MLIR_RegionHandle region, Arena *out) {
    uint32_t r = REGION_IDX(ctx, region, "MLIR_ComputeCFGInfo");
    if (!r || REGIONS(ctx)[r].n_blocks == 0) {
        MLIR_CFGInfo *cfg = arena_new(out, MLIR_CFGInfo);
        memset(cfg, 0, sizeof(*cfg));
        return cfg;
    }
    IR_Region *reg = &REGIONS(ctx)[r];
    size_t n_total = reg->n_blocks;
    // Direct block-index table when the region isn't tiny relative to the whole
    // context; falls back to the hash-map version for tiny functions in huge modules.
    if (ctx->blocks.n > n_total * 4 + 256)
        return mlir_compute_cfg_info_generic(ctx, region, out);

    MLIR_CFGInfo *cfg = arena_new(out, MLIR_CFGInfo);
    memset(cfg, 0, sizeof(*cfg));

    IR_Block *blocks = BLOCKS(ctx);
    IR_Op *ops = OPS(ctx);
    IR_OpExtra *op_extra = OPEXTRA(ctx);
    const uint32_t *extra = EXTRA(ctx);
    const IR_U32Map *emap = &ctx->op_extra_map;

    uint32_t n_ctx_blks = ctx->blocks.n;
    int32_t *dfs_id = arena_new_array(out, int32_t, n_ctx_blks);
    memset(dfs_id, 0xff, (size_t)n_ctx_blks * sizeof(int32_t)); // -1 = unvisited

    uint32_t *stk_blk = arena_new_array(out, uint32_t, n_total);
    const uint32_t **stk_triples = arena_new_array(out, const uint32_t *, n_total);
    int32_t *stk_id = arena_new_array(out, int32_t, n_total);
    uint32_t *stk_succ = arena_new_array(out, uint32_t, n_total);
    uint32_t *stk_ns = arena_new_array(out, uint32_t, n_total);
    uint32_t *po_blk = arena_new_array(out, uint32_t, n_total);
    int32_t *po_id = arena_new_array(out, int32_t, n_total);
    int32_t *dfs_to_rpo = arena_new_array(out, int32_t, n_total);

    size_t edge_cap = n_total * 2 + 8;
    int32_t *edge_u = arena_new_array(out, int32_t, edge_cap);
    int32_t *edge_v = arena_new_array(out, int32_t, edge_cap);
    size_t n_edges = 0;

    uint32_t entry = reg->blocks[0];
    size_t sp = 0, n_po = 0;
    int32_t next_id = 0;
    {
        uint32_t ns = 0;
        const uint32_t *triples = NULL;
        IR_Block *eb = &blocks[entry];
        if (eb->n_ops > 0) {
            uint32_t term = eb->ops[eb->n_ops - 1];
            uint8_t tfl = ops[term].flags;
            if (tfl & OPF_HAS_EXTRA) {
                uint32_t ei = (tfl & OPF_INLINE_EXTRA) ? ops[term].first_result : u32map_get(emap, term);
                const IR_OpExtra *ox = &op_extra[ei];
                ns = ox->n_successors;
                triples = extra + ox->first_extra + ox->n_regions;
            }
        }
        dfs_id[entry] = next_id;
        stk_blk[0] = entry;
        stk_triples[0] = triples;
        stk_id[0] = next_id++;
        stk_succ[0] = 0;
        stk_ns[0] = ns;
        sp = 1;
    }

    while (sp > 0) {
        size_t top = sp - 1;
        uint32_t si = stk_succ[top];
        if (si < stk_ns[top]) {
            stk_succ[top] = si + 1;
            uint32_t succ = stk_triples[top][3 * si];
            if (succ != 0) {
                int32_t cur_id = stk_id[top];
                int32_t sid = dfs_id[succ];
                if (sid < 0) {
                    sid = next_id++;
                    dfs_id[succ] = sid;
                    uint32_t sns = 0;
                    const uint32_t *striples = NULL;
                    IR_Block *sb = &blocks[succ];
                    if (sb->n_ops > 0) {
                        uint32_t sterm = sb->ops[sb->n_ops - 1];
                        uint8_t sfl = ops[sterm].flags;
                        if (sfl & OPF_HAS_EXTRA) {
                            uint32_t ei = (sfl & OPF_INLINE_EXTRA) ? ops[sterm].first_result : u32map_get(emap, sterm);
                            const IR_OpExtra *ox = &op_extra[ei];
                            sns = ox->n_successors;
                            striples = extra + ox->first_extra + ox->n_regions;
                        }
                    }
                    stk_blk[sp] = succ;
                    stk_triples[sp] = striples;
                    stk_id[sp] = sid;
                    stk_succ[sp] = 0;
                    stk_ns[sp] = sns;
                    sp++;
                }
                if (MLIR_NAT_UNLIKELY(n_edges == edge_cap)) {
                    size_t new_cap = edge_cap * 2;
                    int32_t *nu = arena_new_array(out, int32_t, new_cap);
                    int32_t *nv = arena_new_array(out, int32_t, new_cap);
                    memcpy(nu, edge_u, n_edges * sizeof(int32_t));
                    memcpy(nv, edge_v, n_edges * sizeof(int32_t));
                    edge_u = nu;
                    edge_v = nv;
                    edge_cap = new_cap;
                }
                edge_u[n_edges] = cur_id;
                edge_v[n_edges] = sid;
                n_edges++;
            }
        } else {
            sp = top;
            po_blk[n_po] = stk_blk[top];
            po_id[n_po] = stk_id[top];
            n_po++;
        }
    }

    size_t n = n_po;
    if (n == 0) return cfg;

    MLIR_BlockHandle *rpo = arena_new_array(out, MLIR_BlockHandle, n);
    for (size_t i = 0; i < n; i++) {
        size_t ri = n - 1 - i;
        rpo[ri] = BLOCK_H(ctx, po_blk[i]);
        dfs_to_rpo[po_id[i]] = (int32_t)ri;
    }

    uint32_t *pred_off = arena_new_array(out, uint32_t, n + 1);
    memset(pred_off, 0, (n + 1) * sizeof(uint32_t));
    for (size_t e = 0; e < n_edges; e++) {
        int32_t u = dfs_to_rpo[edge_u[e]];
        int32_t v = dfs_to_rpo[edge_v[e]];
        edge_u[e] = u;
        edge_v[e] = v;
        pred_off[(size_t)v + 1]++;
    }
    for (size_t i = 0; i < n; i++) pred_off[i + 1] += pred_off[i];
    int32_t *pred_rpo = arena_new_array(out, int32_t, n_edges > 0 ? n_edges : 1);
    uint32_t *pred_cur = arena_new_array(out, uint32_t, n);
    memcpy(pred_cur, pred_off, n * sizeof(uint32_t));
    for (size_t e = 0; e < n_edges; e++) {
        int32_t v = edge_v[e];
        pred_rpo[pred_cur[v]++] = edge_u[e];
    }

    int32_t *idom = arena_new_array(out, int32_t, n);
    memset(idom, 0xff, n * sizeof(int32_t));
    idom[0] = 0;

    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 1; i < n; i++) {
            uint32_t ps = pred_off[i];
            uint32_t pe = pred_off[i + 1];
            int32_t new_idom;
            if (pe - ps == 1) {
                new_idom = pred_rpo[ps];
                if (idom[new_idom] < 0) continue;
            } else {
                new_idom = -1;
                for (uint32_t p = ps; p < pe; p++) {
                    int32_t pi = pred_rpo[p];
                    if (idom[pi] < 0) continue;
                    if (new_idom < 0) new_idom = pi;
                    else new_idom = nat_cfg_intersect(idom, pi, new_idom);
                }
                if (new_idom < 0) continue;
            }
            if (idom[i] != new_idom) {
                idom[i] = new_idom;
                changed = true;
            }
        }
    }
    idom[0] = -1;

    uint32_t *loop_depth = arena_new_array(out, uint32_t, n);
    bool *is_header = arena_new_array(out, bool, n);
    memset(loop_depth, 0, n * sizeof(uint32_t));
    memset(is_header, 0, n * sizeof(bool));

    cfg->n_blocks = n;
    cfg->rpo = rpo;
    cfg->idom = idom;
    cfg->loop_depth = loop_depth;
    cfg->is_loop_header = is_header;

    int32_t *wl = arena_new_array(out, int32_t, n);
    uint32_t *seen = arena_new_array(out, uint32_t, n);
    memset(seen, 0, n * sizeof(uint32_t));
    uint32_t stamp = 0;

    for (size_t hi = 0; hi < n; hi++) {
        bool found_backedge = false;
        stamp++;
        size_t wn = 0;
        uint32_t pe = pred_off[hi + 1];
        for (uint32_t p = pred_off[hi]; p < pe; p++) {
            int32_t ui = pred_rpo[p];
            if (ui < (int32_t)hi) continue;
            int32_t cur = ui;
            while (cur > (int32_t)hi) cur = idom[cur];
            if (cur == (int32_t)hi) {
                found_backedge = true;
                if (seen[ui] != stamp) {
                    seen[ui] = stamp;
                    wl[wn++] = ui;
                }
            }
        }
        if (!found_backedge) continue;
        is_header[hi] = true;
        seen[hi] = stamp;
        loop_depth[hi]++;
        while (wn > 0) {
            int32_t cur = wl[--wn];
            if (cur == (int32_t)hi) continue;
            loop_depth[cur]++;
            uint32_t cpe = pred_off[cur + 1];
            for (uint32_t p = pred_off[cur]; p < cpe; p++) {
                int32_t pi = pred_rpo[p];
                if (seen[pi] != stamp) {
                    seen[pi] = stamp;
                    wl[wn++] = pi;
                }
            }
        }
    }

    return cfg;
}

bool MLIR_ValueHasOneUse(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_ValueHasOneUse");
    return v != 0 && VALS(ctx)[v].n_uses == 1;
}

bool MLIR_IsValueUnused(MLIR_Context *ctx, MLIR_ValueHandle vh) {
    uint32_t v = VAL_IDX(ctx, vh, "MLIR_IsValueUnused");
    return v == 0 || VALS(ctx)[v].n_uses == 0;
}

MLIR_PredHandle MLIR_GetBlockFirstPred(MLIR_Context *ctx, MLIR_BlockHandle bh) {
    uint32_t b = BLOCK_IDX(ctx, bh, "MLIR_GetBlockFirstPred");
    if (!b) return MLIR_INVALID_HANDLE;
    return (MLIR_PredHandle)BLOCKS(ctx)[b].first_pred;
}

MLIR_PredHandle MLIR_GetPredNext(MLIR_Context *ctx, MLIR_PredHandle pred) {
    API_CHECK(ctx != NULL, "MLIR_GetPredNext", "context is NULL");
    uint32_t p = (uint32_t)pred;
    if (!p) return MLIR_INVALID_HANDLE;
    API_CHECK(p < ctx->preds.n, "MLIR_GetPredNext", "invalid pred handle");
    return (MLIR_PredHandle)PREDS(ctx)[p].next;
}

MLIR_OpHandle MLIR_GetPredTerminator(MLIR_Context *ctx, MLIR_PredHandle pred) {
    API_CHECK(ctx != NULL, "MLIR_GetPredTerminator", "context is NULL");
    uint32_t p = (uint32_t)pred;
    API_CHECK(p > 0 && p < ctx->preds.n, "MLIR_GetPredTerminator", "invalid pred handle");
    return OP_H(ctx, PREDS(ctx)[p].owner);
}

MLIR_BlockHandle MLIR_GetPredBlock(MLIR_Context *ctx, MLIR_PredHandle pred) {
    API_CHECK(ctx != NULL, "MLIR_GetPredBlock", "context is NULL");
    uint32_t p = (uint32_t)pred;
    API_CHECK(p > 0 && p < ctx->preds.n, "MLIR_GetPredBlock", "invalid pred handle");
    return BLOCK_H(ctx, OPS(ctx)[PREDS(ctx)[p].owner].parent_block);
}

size_t MLIR_GetPredSuccessorIndex(MLIR_Context *ctx, MLIR_PredHandle pred) {
    API_CHECK(ctx != NULL, "MLIR_GetPredSuccessorIndex", "context is NULL");
    uint32_t p = (uint32_t)pred;
    API_CHECK(p > 0 && p < ctx->preds.n, "MLIR_GetPredSuccessorIndex", "invalid pred handle");
    return (size_t)PREDS(ctx)[p].succ_idx;
}

// Trait-driven verification of `op` and all ops nested in its regions.
static bool verify_op_recursive(MLIR_Context *ctx, MLIR_OpHandle op) {
    MLIR_OpType ty = MLIR_GetOpType(ctx, op);
    MLIR_OpTraits tr = MLIR_GetOpTypeTraits(ty);
    MLIR_LocationHandle loc = MLIR_GetOpLocation(ctx, op);
    string name = MLIR_GetOpTypeName(ctx, ty);
    size_t n_res = MLIR_GetOpNumResults(ctx, op);
    size_t n_succ = MLIR_GetOpNumSuccessors(ctx, op);
    bool ok = true;

    if ((tr & MLIR_T_ZERO_RESULT) && n_res != 0) {
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, loc,
                            "'%.*s' op expects 0 results, got %zu",
                            (int)name.size, name.str, n_res);
        ok = false;
    }
    if ((tr & MLIR_T_ONE_RESULT) && n_res != 1) {
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, loc,
                            "'%.*s' op expects 1 result, got %zu",
                            (int)name.size, name.str, n_res);
        ok = false;
    }
    if ((tr & MLIR_T_BRANCH) && n_succ == 0) {
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, loc,
                            "'%.*s' op expects at least 1 successor",
                            (int)name.size, name.str);
        ok = false;
    }
    if (ty < OP_TYPE_COUNT && !(tr & MLIR_T_BRANCH) && n_succ > 0) {
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, loc,
                            "'%.*s' op expects 0 successors, got %zu",
                            (int)name.size, name.str, n_succ);
        ok = false;
    }
    if ((tr & MLIR_T_SYMBOL) && MLIR_GetOpAttr(ctx, op, MLIR_KEY_SYM_NAME) == MLIR_INVALID_HANDLE) {
        MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, loc,
                            "'%.*s' op missing required 'sym_name' attribute",
                            (int)name.size, name.str);
        ok = false;
    }

    size_t n_reg = MLIR_GetOpNumRegions(ctx, op);
    for (size_t r = 0; r < n_reg; r++) {
        MLIR_RegionHandle reg = MLIR_GetOpRegion(ctx, op, r);
        size_t n_blk = MLIR_GetRegionNumBlocks(ctx, reg);
        if ((tr & MLIR_T_SINGLE_BLOCK) && n_blk > 1) {
            MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, loc,
                                "'%.*s' op region %zu expects at most 1 block, got %zu",
                                (int)name.size, name.str, r, n_blk);
            ok = false;
        }
        for (size_t b = 0; b < n_blk; b++) {
            MLIR_BlockHandle blk = MLIR_GetRegionBlock(ctx, reg, b);
            size_t n_ops = MLIR_GetBlockNumOps(ctx, blk);
            for (size_t i = 0; i < n_ops; i++) {
                MLIR_OpHandle child = MLIR_GetBlockOp(ctx, blk, i);
                MLIR_OpType cty = MLIR_GetOpType(ctx, child);
                MLIR_OpTraits ctr = MLIR_GetOpTypeTraits(cty);
                if (i + 1 < n_ops && (ctr & MLIR_T_TERMINATOR)) {
                    string cname = MLIR_GetOpTypeName(ctx, cty);
                    MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, MLIR_GetOpLocation(ctx, child),
                                        "terminator '%.*s' must be the last op in its block",
                                        (int)cname.size, cname.str);
                    ok = false;
                }
                if (!verify_op_recursive(ctx, child)) ok = false;
            }
            if (n_ops > 0 && ty < OP_TYPE_COUNT && !(tr & MLIR_T_NO_TERMINATOR)) {
                MLIR_OpHandle last = MLIR_GetBlockOp(ctx, blk, n_ops - 1);
                MLIR_OpType lty = MLIR_GetOpType(ctx, last);
                if (lty < OP_TYPE_COUNT && !(MLIR_GetOpTypeTraits(lty) & MLIR_T_TERMINATOR)) {
                    string lname = MLIR_GetOpTypeName(ctx, lty);
                    MLIR_EmitDiagnostic(ctx, MLIR_DIAG_ERROR, MLIR_GetOpLocation(ctx, last),
                                        "block with no terminator; last op is '%.*s'",
                                        (int)lname.size, lname.str);
                    ok = false;
                }
            }
        }
    }
    return ok;
}

bool MLIR_VerifyOp(MLIR_Context *ctx, MLIR_OpHandle op) {
    API_CHECK(ctx != NULL, "MLIR_VerifyOp", "context is NULL");
    OP_IDX(ctx, op, "MLIR_VerifyOp");
    return verify_op_recursive(ctx, op);
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

static bool pass_should_print_after(string print_after, string name) {
    if (print_after.size == 0) return false;
    return str_eq(print_after, str_lit("*")) || str_eq(print_after, name);
}

bool MLIR_RunPasses(MLIR_Context *ctx, MLIR_OpHandle module,
                    const MLIR_Pass *passes, size_t n,
                    const MLIR_PipelineOptions *opts) {
    API_CHECK(ctx != NULL, "MLIR_RunPasses", "context is NULL");
    API_CHECK(module != MLIR_INVALID_HANDLE, "MLIR_RunPasses", "module handle is NULL");
    if (n == 0) return true;

    const MLIR_TargetInfo *target = opts ? opts->target : NULL;
    bool verify_each = opts ? opts->verify_each : false;
    bool timing = opts ? opts->timing : false;
    string print_after = opts ? opts->print_after : str_lit("");

    Arena *scratch = arena_create(256 * 1024);
    arena_pos_t scratch_base = arena_get_pos(scratch);

    MLIR_PassContext pc;
    pc.ctx = ctx;
    pc.target = target;
    pc.scratch = scratch;

    bool all_ok = true;
    for (size_t i = 0; i < n; i++) {
        const MLIR_Pass *p = &passes[i];
        uint64_t t0 = timing ? platform_clock_ns() : 0;
        bool ok = true;

        if (p->anchor == MLIR_ANCHOR_MODULE) {
            ok = p->run(&pc, module, p->state);
            arena_reset(scratch, scratch_base);
        } else {
            // MLIR_ANCHOR_FUNCTION: visit every op in module with MLIR_T_FUNCTION and a body.
            if (MLIR_GetOpNumRegions(ctx, module) > 0) {
                MLIR_RegionHandle mr = MLIR_GetOpRegion(ctx, module, 0);
                for (MLIR_BlockHandle mb = MLIR_GetRegionFirstBlock(ctx, mr);
                     mb != MLIR_INVALID_HANDLE && ok;
                     mb = MLIR_GetBlockNext(ctx, mb)) {
                    for (MLIR_OpHandle op = MLIR_GetBlockFirstOp(ctx, mb);
                         op != MLIR_INVALID_HANDLE && ok; ) {
                        MLIR_OpHandle next = MLIR_GetOpNext(ctx, op);
                        if (MLIR_OpHasTrait(ctx, op, MLIR_T_FUNCTION) &&
                            MLIR_GetOpNumRegions(ctx, op) > 0 &&
                            MLIR_GetRegionFirstBlock(ctx, MLIR_GetOpRegion(ctx, op, 0)) != MLIR_INVALID_HANDLE) {
                            ok = p->run(&pc, op, p->state);
                            arena_reset(scratch, scratch_base);
                        }
                        op = next;
                    }
                }
            }
        }

        if (!ok) {
            all_ok = false;
            break;
        }
        if (verify_each && !MLIR_VerifyOp(ctx, module)) {
            all_ok = false;
            break;
        }
        if (timing) {
            uint64_t dt_ns = platform_clock_ns() - t0;
            MLIR_EmitDiagnostic(ctx, MLIR_DIAG_REMARK, MLIR_INVALID_HANDLE,
                                "pass %.*s: %.3f ms",
                                (int)p->name.size, p->name.str,
                                (double)dt_ns * 1e-6);
        }
        if (pass_should_print_after(print_after, p->name)) {
            string ir = print_operation_generic(ctx, 0, module);
            MLIR_EmitDiagnostic(ctx, MLIR_DIAG_REMARK, MLIR_INVALID_HANDLE,
                                "// -----// IR Dump After %.*s //----- //\n%.*s",
                                (int)p->name.size, p->name.str,
                                (int)ir.size, ir.str);
        }
    }

    arena_destroy(scratch);
    return all_ok;
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

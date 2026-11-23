// IR storage benchmark: the API every storage design implements.
//
// One workload (workload.c) drives four implementations of a minimal IR core
// through this header:
//
//   ir_a_today.c  A, pointers into an arena, today's layout. A close copy of
//                 IR_Op/IR_OpAux/IR_Value/IR_ValueAux/UseNode/IR_Block/
//                 IR_Region in mlir_api_impl.c and of what MLIR_CreateOp*,
//                 MLIR_EraseOp, MLIR_ReplaceAllUsesOfValue, ... do.
//   ir_a_lean.c   A, pointers into an arena, compact layout: one allocation
//                 per op holding the op header and its operands (with intrusive
//                 use links, like upstream's OpOperand), results, attributes,
//                 regions and successors (design doc §6 N5/N6).
//   ir_b_pool.c   B, 32-bit indices into context-owned vectors that double by
//                 reallocation. Operands live in one shared operand vector
//                 (each op stores start + count); use links are operand
//                 indices; results are consecutive entries of the value vector.
//                 With -DB_SEGMENTED (B-pool-seg) each vector is a table of
//                 fixed-size arena segments instead, so it never reallocates.
//   ir_b_arena.c  B, 32-bit indices for ops/values/blocks/regions, but each
//                 op's operand list is an arena array of 32-bit value indices,
//                 and each value keeps an arena array of its uses.
//
// Handles are IR_HANDLE_BITS wide: 64 for the pointer designs (the handle is
// the object's address), 32 for the index designs (0 is invalid). Every
// function takes the context, which the pointer designs ignore.
#ifndef IR_STORAGE_IR_API_H
#define IR_STORAGE_IR_API_H

#include <stdint.h>
#include <stddef.h>

#ifndef IR_HANDLE_BITS
#error "define IR_HANDLE_BITS (64 for the A designs, 32 for the B designs)"
#endif

#if IR_HANDLE_BITS == 64
typedef uint64_t IrHandle;
#else
typedef uint32_t IrHandle;
#endif

typedef IrHandle IrOp;
typedef IrHandle IrValue;
typedef IrHandle IrBlock;
typedef IrHandle IrRegion;
typedef IrHandle IrType;
typedef IrHandle IrAttr;
// A use (one operand slot). Always 64 bits: the B-arena design packs
// (value, position) into it.
typedef uint64_t IrUse;

#define IR_INVALID 0

typedef struct IrCtx IrCtx;

// Op kinds, shaped like tinyc's LLVM-dialect output.
enum {
    OPK_MODULE = 1,
    OPK_FUNC,
    OPK_CONST,
    OPK_GEP,
    OPK_LOAD,
    OPK_STORE,
    OPK_ICMP,
    OPK_ADD,
    OPK_SUB,
    OPK_MUL,
    OPK_SEXT,
    OPK_SELECT,
    OPK_CALL,
    OPK_ALLOCA,
    OPK_BR,
    OPK_COND_BR,
    OPK_RETURN,
    OPK_COUNT
};

// Attribute names. A-today stores the name as a string it copies into the
// arena (as MLIR_CreateAttribute* do); the other designs store the key.
enum {
    ATTR_VALUE = 1,
    ATTR_PREDICATE,
    ATTR_CALLEE,
    ATTR_ELEM_TYPE,
    ATTR_SYM_NAME,
    ATTR_FUNC_TYPE
};

// Everything needed to create an op. Operands are flat: the regular operands
// first, then each successor's operands in successor order (D5).
typedef struct IrOpSpec {
    uint32_t op_type;
    const IrValue *operands;
    uint32_t n_operands;          // regular operands only
    const IrType *result_types;
    uint32_t n_results;
    const IrAttr *attrs;
    uint32_t n_attrs;
    const IrRegion *regions;
    uint32_t n_regions;
    const IrBlock *succs;
    uint32_t n_succs;
    const uint32_t *n_succ_operands;  // per successor
} IrOpSpec;

const char *ir_impl_name(void);

IrCtx *ir_ctx_create(void);
void ir_ctx_destroy(IrCtx *c);
// Size hint: the flat-vector B designs reserve their vectors for about this
// many ops, so they rarely reallocate while building (the A designs and
// B-pool-seg ignore it).
void ir_ctx_reserve(IrCtx *c, uint32_t n_ops);
// Bytes of IR storage in use: arena bytes handed out plus vector elements in
// use (not vector capacity; slack and freed blocks show in peak RSS).
uint64_t ir_ctx_bytes(IrCtx *c);

IrType ir_type_int(IrCtx *c, uint32_t width);
IrType ir_type_ptr(IrCtx *c);
// Bit width of an integer type, 0 for the pointer type.
uint32_t ir_type_width(IrCtx *c, IrType t);

IrAttr ir_attr_int(IrCtx *c, uint32_t key, int64_t value, IrType type);
int64_t ir_attr_int_value(IrCtx *c, IrAttr a);

IrRegion ir_region_create(IrCtx *c);
void ir_region_append_block(IrCtx *c, IrRegion r, IrBlock b);
uint32_t ir_region_num_blocks(IrCtx *c, IrRegion r);
IrBlock ir_region_block(IrCtx *c, IrRegion r, uint32_t i);

IrBlock ir_block_create(IrCtx *c);
IrValue ir_block_add_arg(IrCtx *c, IrBlock b, IrType t);
uint32_t ir_block_num_args(IrCtx *c, IrBlock b);
IrValue ir_block_arg(IrCtx *c, IrBlock b, uint32_t i);
uint32_t ir_block_num_ops(IrCtx *c, IrBlock b);
IrOp ir_block_op(IrCtx *c, IrBlock b, uint32_t i);
void ir_block_append_op(IrCtx *c, IrBlock b, IrOp op);
void ir_block_insert_op(IrCtx *c, IrBlock b, uint32_t idx, IrOp op);

IrOp ir_op_create(IrCtx *c, const IrOpSpec *s);
uint32_t ir_op_type(IrCtx *c, IrOp op);
uint32_t ir_op_num_operands(IrCtx *c, IrOp op);
IrValue ir_op_operand(IrCtx *c, IrOp op, uint32_t i);
uint32_t ir_op_num_results(IrCtx *c, IrOp op);
IrValue ir_op_result(IrCtx *c, IrOp op, uint32_t i);
uint32_t ir_op_num_attrs(IrCtx *c, IrOp op);
IrAttr ir_op_attr(IrCtx *c, IrOp op, uint32_t i);
uint32_t ir_op_num_regions(IrCtx *c, IrOp op);
IrRegion ir_op_region(IrCtx *c, IrOp op, uint32_t i);
uint32_t ir_op_num_succs(IrCtx *c, IrOp op);
IrBlock ir_op_succ(IrCtx *c, IrOp op, uint32_t s);
uint32_t ir_op_num_succ_operands(IrCtx *c, IrOp op, uint32_t s);
IrValue ir_op_succ_operand(IrCtx *c, IrOp op, uint32_t s, uint32_t i);
IrBlock ir_op_parent_block(IrCtx *c, IrOp op);

IrType ir_value_type(IrCtx *c, IrValue v);
// The defining op of an op result; IR_INVALID for a block argument.
IrOp ir_value_def_op(IrCtx *c, IrValue v);
uint32_t ir_value_num_uses(IrCtx *c, IrValue v);
// Use iteration; 0 ends the list. The operand index is flat: regular
// operands first, then successor operands.
IrUse ir_value_first_use(IrCtx *c, IrValue v);
IrUse ir_use_next(IrCtx *c, IrUse u);
IrOp ir_use_owner(IrCtx *c, IrUse u);
uint32_t ir_use_operand_index(IrCtx *c, IrUse u);

// `idx` is a flat operand index (as ir_use_operand_index returns).
void ir_op_set_operand(IrCtx *c, IrOp op, uint32_t idx, IrValue v);
void ir_replace_all_uses(IrCtx *c, IrValue from, IrValue to);
// Unlinks the op's operands from their use lists and removes it from its
// block. The op's results must have no uses. The handle stays readable
// (an empty slot in the B designs).
void ir_op_erase(IrCtx *c, IrOp op);

#endif

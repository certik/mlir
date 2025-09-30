#include "mlir_api.h"

// This file provides an example implementation of the public C API on top of
// the upstream MLIR C++ library.  It is not compiled as part of this project
// but demonstrates how the API can be bridged to the official MLIR data
// structures.

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>

using namespace mlir;

// Global context/builder used to create operations.  A real implementation
// would likely provide a more sophisticated ownership model.
static MLIRContext *gContext = nullptr;
static OpBuilder *gBuilder = nullptr;

// Mapping from MLIR operation pointers back to our OpType enumeration.
static llvm::DenseMap<const Operation *, OpType> gOpTypeMap;

// Mapping from OpType to the canonical MLIR operation name.  The order matches
// the OpType enumeration defined in mlir_api.h.
static const char *const kOpNames[OP_TYPE_COUNT] = {
    "",                     // OP_TYPE_UNREGISTERED
    "builtin.module",       // OP_TYPE_MODULE
    "arith.addi",           // OP_TYPE_ARITH_ADDI
    "arith.subi",           // OP_TYPE_ARITH_SUBI
    "arith.muli",           // OP_TYPE_ARITH_MULI
    "arith.divi",           // OP_TYPE_ARITH_DIVI
    "arith.addf",           // OP_TYPE_ARITH_ADDF
    "arith.subf",           // OP_TYPE_ARITH_SUBF
    "arith.mulf",           // OP_TYPE_ARITH_MULF
    "arith.divf",           // OP_TYPE_ARITH_DIVF
    "arith.constant",       // OP_TYPE_ARITH_CONSTANT
    "arith.cmpi",           // OP_TYPE_ARITH_CMPI
    "arith.cmpf",           // OP_TYPE_ARITH_CMPF
    "arith.select",         // OP_TYPE_ARITH_SELECT
    "memref.load",          // OP_TYPE_MEMREF_LOAD
    "memref.store",         // OP_TYPE_MEMREF_STORE
    "memref.alloc",         // OP_TYPE_MEMREF_ALLOC
    "memref.dealloc",       // OP_TYPE_MEMREF_DEALLOC
    "cf.br",                // OP_TYPE_CF_BR
    "cf.cond_br",           // OP_TYPE_CF_COND_BR
    "cf.switch",            // OP_TYPE_CF_SWITCH
    "func.func",            // OP_TYPE_FUNC_FUNC
    "func.return",          // OP_TYPE_FUNC_RETURN
    "func.call",            // OP_TYPE_FUNC_CALL
    "scf.for",              // OP_TYPE_SCF_FOR
    "scf.while",            // OP_TYPE_SCF_WHILE
    "scf.if",               // OP_TYPE_SCF_IF
    "scf.yield",            // OP_TYPE_SCF_YIELD
    "tt.get_program_id",    // OP_TYPE_TT_GET_PROGRAM_ID
    "tt.load",              // OP_TYPE_TT_LOAD
    "tt.store",             // OP_TYPE_TT_STORE
    "tt.make_range",        // OP_TYPE_TT_MAKE_RANGE
    "tt.splat",             // OP_TYPE_TT_SPLAT
    "tt.addptr",            // OP_TYPE_TT_ADDPTR
    "tt.return",            // OP_TYPE_TT_RETURN
    "tt.func",              // OP_TYPE_TT_FUNC
    "tt.call",              // OP_TYPE_TT_CALL
    "tt.reduce",            // OP_TYPE_TT_REDUCE
    "gpu.launch",           // OP_TYPE_GPU_LAUNCH
    "affine.for",           // OP_TYPE_AFFINE_FOR
    "affine.load",          // OP_TYPE_AFFINE_LOAD
    "vector.print",         // OP_TYPE_VECTOR_PRINT
    "std.constant",         // OP_TYPE_STD_CONSTANT
    "std.return",           // OP_TYPE_STD_RETURN
    "tensor.extract",       // OP_TYPE_TENSOR_EXTRACT
    "tensor.splat",         // OP_TYPE_TENSOR_SPLAT
    "tensor.collapse_shape",// OP_TYPE_TENSOR_COLLAPSE_SHAPE
    "linalg.fill",          // OP_TYPE_LINALG_FILL
    "index.constant",       // OP_TYPE_INDEX_CONSTANT
    "return",               // OP_TYPE_RETURN
    "tt.reduce_return"      // OP_TYPE_TT_REDUCE_RETURN
};

static inline StringRef opTypeToName(OpType type) {
    assert(type < OP_TYPE_COUNT);
    return kOpNames[type];
}

#ifdef __cplusplus
extern "C" {
#endif

static OpType lookupOpTypeByName(StringRef name) {
    for (int i = 1; i < OP_TYPE_COUNT; ++i) {
        if (name == opTypeToName(static_cast<OpType>(i)))
            return static_cast<OpType>(i);
    }
    return OP_TYPE_UNREGISTERED;
}

void mlir_api_init(MlirOperation *root) {
    if (!gContext) {
        gContext = new MLIRContext();
        gContext->loadAllAvailableDialects();
        gBuilder = new OpBuilder(gContext);
    }

    gOpTypeMap.clear();

    if (root) {
        Operation *cppRoot = reinterpret_cast<Operation *>(root);
        cppRoot->walk([](Operation *op) {
            gOpTypeMap[op] = lookupOpTypeByName(op->getName().getStringRef());
        });
    }
}

MlirOperation *mlir_op_create(Arena *arena, OpType type) {
    (void)arena;
    OperationState state(gBuilder->getUnknownLoc(), opTypeToName(type));
    Operation *op = Operation::create(state);
    gOpTypeMap[op] = type;
    return reinterpret_cast<MlirOperation *>(op);
}

void mlir_op_add_region(Arena *arena, MlirOperation *op, MlirRegion *region) {
    (void)arena;
    Operation *cppOp = reinterpret_cast<Operation *>(op);
    Region *cppRegion = reinterpret_cast<Region *>(region);
    cppOp->getRegions().push_back(cppRegion);
}

void mlir_op_add_operand(Arena *arena, MlirOperation *op, MlirValue *operand) {
    (void)arena;
    Operation *cppOp = reinterpret_cast<Operation *>(op);
    Value cppVal = *reinterpret_cast<Value *>(operand);
    cppOp->insertOperands(cppOp->getNumOperands(), cppVal);
}

void mlir_op_add_result(Arena *arena, MlirOperation *op, MlirValue *result) {
    (void)arena;
    Operation *cppOp = reinterpret_cast<Operation *>(op);
    Value cppVal = *reinterpret_cast<Value *>(result);
    cppOp->insertResults(cppOp->getNumResults(), cppVal.getType());
}

void mlir_block_add_operation(Arena *arena, MlirBlock *block, MlirOperation *op) {
    (void)arena;
    Block *cppBlock = reinterpret_cast<Block *>(block);
    Operation *cppOp = reinterpret_cast<Operation *>(op);
    cppBlock->push_back(cppOp);
}

void mlir_block_add_argument(Arena *arena, MlirBlock *block, MlirValue *arg) {
    (void)arena;
    Block *cppBlock = reinterpret_cast<Block *>(block);
    Value cppVal = *reinterpret_cast<Value *>(arg);
    cppBlock->addArgument(cppVal.getType(), cppVal.getLoc());
}

void mlir_region_add_block(Arena *arena, MlirRegion *region, MlirBlock *block) {
    (void)arena;
    Region *cppRegion = reinterpret_cast<Region *>(region);
    Block *cppBlock = reinterpret_cast<Block *>(block);
    cppRegion->push_back(cppBlock);
}

size_t mlir_region_num_blocks(const MlirRegion *region) {
    const Region *cppRegion = reinterpret_cast<const Region *>(region);
    return cppRegion->getBlocks().size();
}

MlirBlock *mlir_region_get_block(const MlirRegion *region, size_t idx) {
    Region *cppRegion = const_cast<Region *>(reinterpret_cast<const Region *>(region));
    auto it = cppRegion->begin();
    std::advance(it, idx);
    return reinterpret_cast<MlirBlock *>(&*it);
}

size_t mlir_block_num_operations(const MlirBlock *block) {
    const Block *cppBlock = reinterpret_cast<const Block *>(block);
    return cppBlock->getOperations().size();
}

MlirOperation *mlir_block_get_operation(const MlirBlock *block, size_t idx) {
    Block *cppBlock = const_cast<Block *>(reinterpret_cast<const Block *>(block));
    auto it = cppBlock->begin();
    std::advance(it, idx);
    return reinterpret_cast<MlirOperation *>(&*it);
}

OpType mlir_operation_get_type(const MlirOperation *op) {
    const Operation *cppOp = reinterpret_cast<const Operation *>(op);
    auto it = gOpTypeMap.find(cppOp);
    if (it != gOpTypeMap.end())
        return it->second;
    return OP_TYPE_UNREGISTERED;
}

size_t mlir_operation_num_regions(const MlirOperation *op) {
    const Operation *cppOp = reinterpret_cast<const Operation *>(op);
    return cppOp->getNumRegions();
}

MlirRegion *mlir_operation_get_region(const MlirOperation *op, size_t idx) {
    Operation *cppOp = const_cast<Operation *>(reinterpret_cast<const Operation *>(op));
    return reinterpret_cast<MlirRegion *>(&cppOp->getRegion(idx));
}

void mlir_operation_set_source_line_start(MlirOperation *op, int64_t line_start) {
    (void)op;
    (void)line_start;
}

void mlir_operation_set_unnumbered_loc_def(MlirOperation *op, MlirLocation *loc) {
    (void)op;
    (void)loc;
}

MlirLocation *mlir_location_create(Arena *arena) {
    (void)arena;
    return nullptr;
}

void mlir_location_set_kind(MlirLocation *loc, MlirLocationKind kind) {
    (void)loc;
    (void)kind;
}

void mlir_location_set_original_text(MlirLocation *loc, string text) {
    (void)loc;
    (void)text;
}

void mlir_location_set_file_data(MlirLocation *loc, string filename, int line, int column) {
    (void)loc;
    (void)filename;
    (void)line;
    (void)column;
}

void mlir_location_set_name_data(MlirLocation *loc, string name) {
    (void)loc;
    (void)name;
}

void mlir_location_set_ref_id(MlirLocation *loc, int ref_id) {
    (void)loc;
    (void)ref_id;
}

void mlir_value_set_location(MlirValue *value, MlirLocation *loc) {
    (void)value;
    (void)loc;
}

void mlir_value_set_divisibility(MlirValue *value, bool has_value, int64_t div_value, MlirType *type) {
    (void)value;
    (void)has_value;
    (void)div_value;
    (void)type;
}

void mlir_value_set_max_divisibility(MlirValue *value, bool has_value, int64_t div_value, MlirType *type) {
    (void)value;
    (void)has_value;
    (void)div_value;
    (void)type;
}

#ifdef __cplusplus
} // extern "C"
#endif

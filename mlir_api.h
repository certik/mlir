// Public MLIR C API declarations without exposing internal data structures.
// Implementations can back these APIs with different MLIR representations.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <base/arena.h>
#include <base/string.h>
#include <base/vector.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Handle types - opaque integer handles replacing raw pointers
// -----------------------------------------------------------------------------

typedef uintptr_t MLIR_OpHandle;
typedef uintptr_t MLIR_RegionHandle;
typedef uintptr_t MLIR_BlockHandle;
typedef uintptr_t MLIR_ValueHandle;
typedef uintptr_t MLIR_TypeHandle;
typedef uintptr_t MLIR_AttributeHandle;
typedef uintptr_t MLIR_LocationHandle;

#define MLIR_INVALID_HANDLE 0

typedef struct MLIR_Context {
    Arena *arena;
} MLIR_Context;

typedef struct MLIR_LocationMap {
    void *impl;
} MLIR_LocationMap;

size_t MLIR_GetLocationMapSize(const MLIR_LocationMap *location_map);
size_t MLIR_CollectLocationMap(const MLIR_LocationMap *location_map, string *out_keys, MLIR_LocationHandle *out_locs, size_t max);

// Vector helpers used by the parser implementation
DEFINE_VECTOR_FOR_TYPE(MLIR_OpHandle, VecOp)
DEFINE_VECTOR_FOR_TYPE(MLIR_ValueHandle, VecValue)
DEFINE_VECTOR_FOR_TYPE(MLIR_BlockHandle, VecBlock)
DEFINE_VECTOR_FOR_TYPE(MLIR_AttributeHandle, VecAttribute)

// -----------------------------------------------------------------------------
// Operation kinds
// -----------------------------------------------------------------------------

typedef enum {
    // Core ops
    OP_TYPE_UNREGISTERED = 0,
    OP_TYPE_MODULE,

    // Arithmetic dialect
    OP_TYPE_ARITH_ADDI,
    OP_TYPE_ARITH_SUBI,
    OP_TYPE_ARITH_MULI,
    OP_TYPE_ARITH_DIVI,
    OP_TYPE_ARITH_ADDF,
    OP_TYPE_ARITH_SUBF,
    OP_TYPE_ARITH_MULF,
    OP_TYPE_ARITH_DIVF,
    OP_TYPE_ARITH_CONSTANT,
    OP_TYPE_ARITH_CMPI,
    OP_TYPE_ARITH_CMPF,
    OP_TYPE_ARITH_SELECT,
    OP_TYPE_ARITH_BITCAST,
    OP_TYPE_ARITH_SITOFP,
    OP_TYPE_ARITH_EXTSI,
    OP_TYPE_ARITH_TRUNCI,
    OP_TYPE_ARITH_EXTF,
    OP_TYPE_ARITH_TRUNCF,
    OP_TYPE_ARITH_EXTUI,
    OP_TYPE_ARITH_MAXF,
    OP_TYPE_ARITH_DIVSI,
    OP_TYPE_ARITH_REMSI,
    OP_TYPE_ARITH_ORI,
    OP_TYPE_ARITH_MINSI,
    OP_TYPE_ARITH_ANDI,

    // Math dialect
    OP_TYPE_MATH_EXP,
    OP_TYPE_MATH_LOG,

    // Memory dialect
    OP_TYPE_MEMREF_LOAD,
    OP_TYPE_MEMREF_STORE,
    OP_TYPE_MEMREF_ALLOC,
    OP_TYPE_MEMREF_DEALLOC,

    // Control flow
    OP_TYPE_CF_BR,
    OP_TYPE_CF_COND_BR,
    OP_TYPE_CF_SWITCH,

    // Function dialect
    OP_TYPE_FUNC_FUNC,
    OP_TYPE_FUNC_RETURN,
    OP_TYPE_FUNC_CALL,

    // SCF dialect
    OP_TYPE_SCF_FOR,
    OP_TYPE_SCF_WHILE,
    OP_TYPE_SCF_IF,
    OP_TYPE_SCF_YIELD,

    // Triton dialect
    OP_TYPE_TT_GET_PROGRAM_ID,
    OP_TYPE_TT_LOAD,
    OP_TYPE_TT_STORE,
    OP_TYPE_TT_MAKE_RANGE,
    OP_TYPE_TT_SPLAT,
    OP_TYPE_TT_ADDPTR,
    OP_TYPE_TT_RETURN,
    OP_TYPE_TT_FUNC,
    OP_TYPE_TT_CALL,
    OP_TYPE_TT_REDUCE,
    OP_TYPE_TT_BROADCAST,
    OP_TYPE_TT_EXPAND_DIMS,
    OP_TYPE_TT_DOT,
    OP_TYPE_TT_PURE_EXTERN_ELEMENTWISE,

    // GPU dialect
    OP_TYPE_GPU_LAUNCH,

    // Affine dialect
    OP_TYPE_AFFINE_FOR,
    OP_TYPE_AFFINE_LOAD,

    // Vector dialect
    OP_TYPE_VECTOR_PRINT,

    // Standard dialect
    OP_TYPE_STD_CONSTANT,
    OP_TYPE_STD_RETURN,

    // Tensor dialect
    OP_TYPE_TENSOR_EXTRACT,
    OP_TYPE_TENSOR_SPLAT,
    OP_TYPE_TENSOR_COLLAPSE_SHAPE,

    // Linalg dialect
    OP_TYPE_LINALG_FILL,

    // Index dialect
    OP_TYPE_INDEX_CONSTANT,

    // LLVM dialect
    OP_TYPE_LLVM_MLIR_UNDEF,

    // Return operations
    OP_TYPE_RETURN,
    OP_TYPE_TT_REDUCE_RETURN,

    OP_TYPE_COUNT
} MLIR_OpType;

string MLIR_MLIR_OpTypeToString(MLIR_OpType type);

// -----------------------------------------------------------------------------
// API lifecycle
// -----------------------------------------------------------------------------

void MLIR_InitApi(MLIR_Context *ctx, MLIR_OpHandle root);
void MLIR_SetArenaAllocator(MLIR_Context *ctx, Arena *arena);
Arena *MLIR_GetArenaAllocator(MLIR_Context *ctx);

// -----------------------------------------------------------------------------
// Operation API
// -----------------------------------------------------------------------------

// Creation & structural mutation
MLIR_OpHandle MLIR_CreateOp(
    MLIR_Context *ctx,
    MLIR_OpType type,
    string opname,
    MLIR_AttributeHandle *attributes, size_t n_attributes,
    MLIR_TypeHandle *result_types, size_t n_result_types,
    MLIR_ValueHandle *results, size_t n_results,
    MLIR_ValueHandle *operands, size_t n_operands,
    MLIR_RegionHandle *regions, size_t n_regions,
    MLIR_LocationHandle location,
    MLIR_LocationHandle unnumbered_loc_def,
    string trailing_comment,
    int64_t source_line_start);
void MLIR_AppendOpAttribute(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_AttributeHandle attr);

// Print an operation using upstream MLIR's pretty-printer in generic form
// (mlir::OpPrintingFlags().printGenericOpForm()). Only meaningful when the
// implementation is the upstream-MLIR backend; the native impl returns an
// empty string.
string MLIR_PrintOperationUpstream(MLIR_Context *ctx, MLIR_OpHandle op);

// Accessors
MLIR_OpType MLIR_GetOpType(MLIR_OpHandle op);
string MLIR_GetOpName(MLIR_OpHandle op);
string MLIR_GetOpName_string(MLIR_OpHandle op);
MLIR_LocationHandle MLIR_GetOpLocation(MLIR_OpHandle op);
string MLIR_GetOpTrailingComment(MLIR_OpHandle op);
int64_t MLIR_GetOpSourceLineStart(MLIR_OpHandle op);
MLIR_LocationHandle MLIR_GetOpUnnumberedLocationDef(MLIR_OpHandle op);
size_t MLIR_GetOpNumOperands(MLIR_OpHandle op);
MLIR_ValueHandle MLIR_GetOpOperand(MLIR_OpHandle op, size_t idx);
size_t MLIR_GetOpNumResults(MLIR_OpHandle op);
MLIR_ValueHandle MLIR_GetOpResult(MLIR_OpHandle op, size_t idx);
size_t MLIR_GetOpNumResultTypes(MLIR_OpHandle op);
MLIR_TypeHandle MLIR_GetOpResult_type(MLIR_OpHandle op, size_t idx);
size_t MLIR_GetOpNumAttributes(MLIR_OpHandle op);
MLIR_AttributeHandle MLIR_GetOpAttribute(MLIR_OpHandle op, size_t idx);
size_t MLIR_GetOpNumRegions(MLIR_OpHandle op);
MLIR_RegionHandle MLIR_GetOpRegion(MLIR_OpHandle op, size_t idx);

// -----------------------------------------------------------------------------
// Region API
// -----------------------------------------------------------------------------

MLIR_RegionHandle MLIR_CreateRegion(MLIR_Context *ctx);
void MLIR_AppendRegionBlock(MLIR_Context *ctx, MLIR_RegionHandle region, MLIR_BlockHandle block);
size_t MLIR_GetRegionNumBlocks(MLIR_RegionHandle region);
MLIR_BlockHandle MLIR_GetRegionBlock(MLIR_RegionHandle region, size_t idx);

// -----------------------------------------------------------------------------
// Block API
// -----------------------------------------------------------------------------

MLIR_BlockHandle MLIR_CreateBlock(MLIR_Context *ctx);
void MLIR_AppendBlockOp(MLIR_Context *ctx, MLIR_BlockHandle block, MLIR_OpHandle op);
void MLIR_AppendBlockArg(MLIR_Context *ctx, MLIR_BlockHandle block, MLIR_ValueHandle arg);
size_t MLIR_GetBlockNumOps(MLIR_BlockHandle block);
MLIR_OpHandle MLIR_GetBlockOp(MLIR_BlockHandle block, size_t idx);
size_t MLIR_GetBlockNumArgs(MLIR_BlockHandle block);
MLIR_ValueHandle MLIR_GetBlockArg(MLIR_BlockHandle block, size_t idx);

// -----------------------------------------------------------------------------
// Value API
// -----------------------------------------------------------------------------

// Value kind used for SSA values
typedef enum MLIR_ValueKind {
    BLOCK_ARG,
    OP_RESULT
} MLIR_ValueKind;

// Creation
MLIR_ValueHandle MLIR_CreateValueBlockArg(MLIR_Context *ctx, string register_name, uint32_t result_index, MLIR_TypeHandle type, MLIR_LocationHandle location);
MLIR_ValueHandle MLIR_CreateValueOpResult(MLIR_Context *ctx, MLIR_OpHandle def, uint32_t result_index, MLIR_TypeHandle type, string register_name, MLIR_LocationHandle location);

// Accessors
MLIR_ValueKind MLIR_GetValueKind(MLIR_ValueHandle value);
MLIR_TypeHandle MLIR_GetValueType(MLIR_ValueHandle value);
string MLIR_GetValueRegisterName(MLIR_ValueHandle value);
uint32_t MLIR_GetValueResultIndex(MLIR_ValueHandle value);
MLIR_OpHandle MLIR_GetValueDefiningOp(MLIR_ValueHandle value);
MLIR_LocationHandle MLIR_GetValueLocation(MLIR_ValueHandle value);

// -----------------------------------------------------------------------------
// Type API
// -----------------------------------------------------------------------------

// Creation
MLIR_TypeHandle MLIR_CreateTypeInteger(MLIR_Context *ctx, uint32_t width, bool is_signed);
MLIR_TypeHandle MLIR_CreateTypeFloat(MLIR_Context *ctx, uint32_t width, bool is_bfloat);
MLIR_TypeHandle MLIR_CreateTypeIndex(MLIR_Context *ctx);
MLIR_TypeHandle MLIR_CreateTypeUnknown(MLIR_Context *ctx);
MLIR_TypeHandle MLIR_CreateTypeTensor(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type);
MLIR_TypeHandle MLIR_CreateTypeMemref(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type);
MLIR_TypeHandle MLIR_CreateTypePointer(MLIR_Context *ctx, MLIR_TypeHandle element_type, bool has_address_space, uint32_t address_space);
MLIR_TypeHandle MLIR_CreateTypeOpaque(MLIR_Context *ctx, string name);

// Mutation
void MLIR_SetTypeIntegerProperties(MLIR_TypeHandle type, uint32_t width, bool is_signed);
void MLIR_SetTypeFloatProperties(MLIR_TypeHandle type, uint32_t width, bool is_bfloat);
void MLIR_SetTypeTensorProperties(MLIR_TypeHandle type, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type);
void MLIR_SetTypeMemrefProperties(MLIR_TypeHandle type, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type);
void MLIR_SetTypePointerProperties(MLIR_TypeHandle type, MLIR_TypeHandle element_type, bool has_address_space, uint32_t address_space);

// Introspection & formatting
bool MLIR_IsTypeInteger(MLIR_TypeHandle type);
bool MLIR_IsTypeFloat(MLIR_TypeHandle type);
bool MLIR_IsTypeTensor(MLIR_TypeHandle type);
bool MLIR_IsTypeMemref(MLIR_TypeHandle type);
bool MLIR_IsTypePointer(MLIR_TypeHandle type);
bool MLIR_IsTypeIndex(MLIR_TypeHandle type);
bool MLIR_IsTypeUnknown(MLIR_TypeHandle type);
bool MLIR_IsTypeOpaque(MLIR_TypeHandle type);
string MLIR_GetTypeString(MLIR_Context *ctx, MLIR_TypeHandle type);

// -----------------------------------------------------------------------------
// Attribute API
// -----------------------------------------------------------------------------

// Creation & mutation
MLIR_AttributeHandle MLIR_CreateAttributeInteger(MLIR_Context *ctx, string name, int64_t value);
MLIR_AttributeHandle MLIR_CreateAttributeFloat(MLIR_Context *ctx, string name, double value);
MLIR_AttributeHandle MLIR_CreateAttributeBool(MLIR_Context *ctx, string name, bool value);
MLIR_AttributeHandle MLIR_CreateAttributeString(MLIR_Context *ctx, string name, string value);
MLIR_AttributeHandle MLIR_CreateAttributeArray(MLIR_Context *ctx, string name, MLIR_AttributeHandle *elements, size_t count);
MLIR_AttributeHandle MLIR_CreateAttributeDict(MLIR_Context *ctx, string name, MLIR_AttributeHandle *elements, size_t count);

// Introspection
typedef enum {
    MLIR_ATTR_KIND_INTEGER,
    MLIR_ATTR_KIND_FLOAT,
    MLIR_ATTR_KIND_STRING,
    MLIR_ATTR_KIND_BOOL,
    MLIR_ATTR_KIND_ARRAY,
    MLIR_ATTR_KIND_DICT,
    MLIR_ATTR_KIND_OTHER
} MLIR_AttrKind;

MLIR_AttrKind MLIR_GetAttributeKind(MLIR_AttributeHandle attr);
string MLIR_GetAttributeName(MLIR_AttributeHandle attr);
// Print any attribute as a string (e.g. "0 : i64", "@callee", "[1, 2]").
// Used as a fallback when the kind is MLIR_ATTR_KIND_OTHER.
string MLIR_GetAttributeAsString(MLIR_Context *ctx, MLIR_AttributeHandle attr);
int64_t MLIR_GetAttributeInteger(MLIR_AttributeHandle attr);
double MLIR_GetAttributeFloat(MLIR_AttributeHandle attr);
bool MLIR_GetAttributeBool(MLIR_AttributeHandle attr);
string MLIR_GetAttributeString(MLIR_AttributeHandle attr);
size_t MLIR_GetAttributeArraySize(MLIR_AttributeHandle attr);
MLIR_AttributeHandle MLIR_GetAttributeArrayElement(MLIR_AttributeHandle attr, size_t idx);
size_t MLIR_GetAttributeDictSize(MLIR_AttributeHandle attr);
MLIR_AttributeHandle MLIR_GetAttributeDictElement(MLIR_AttributeHandle attr, size_t idx);

// -----------------------------------------------------------------------------
// Location API
// -----------------------------------------------------------------------------

typedef enum {
    MLIR_LOC_UNKNOWN,
    MLIR_LOC_FILE,
    MLIR_LOC_NAME,
    MLIR_LOC_CALLSITE,
    MLIR_LOC_FUSED,
    MLIR_LOC_REF
} MLIR_LocationKind;

// Creation
MLIR_LocationHandle MLIR_CreateLocationUnknown(MLIR_Context *ctx, string original_text);
MLIR_LocationHandle MLIR_CreateLocationFile(MLIR_Context *ctx, string filename, int line, int column);
MLIR_LocationHandle MLIR_CreateLocationName(MLIR_Context *ctx, string name);
MLIR_LocationHandle MLIR_CreateLocationRef(MLIR_Context *ctx, int ref_id);

// Accessors
MLIR_LocationKind MLIR_GetLocationKind(MLIR_LocationHandle loc);
string MLIR_GetLocationOriginalText(MLIR_LocationHandle loc);
string MLIR_GetLocationFileFilename(MLIR_LocationHandle loc);
int MLIR_GetLocationFileLine(MLIR_LocationHandle loc);
int MLIR_GetLocationFileColumn(MLIR_LocationHandle loc);
string MLIR_GetLocationName(MLIR_LocationHandle loc);
int MLIR_GetLocationRefId(MLIR_LocationHandle loc);

#ifdef __cplusplus
}
#endif

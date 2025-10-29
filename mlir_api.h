// Public MLIR C API declarations without exposing internal data structures.
// Implementations can back these APIs with different MLIR representations.

#pragma once

#include <stddef.h>

#include <base/arena.h>
#include <base/string.h>
#include <base/vector.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Opaque forward declarations
// -----------------------------------------------------------------------------

typedef struct MLIR_Op MLIR_Op;
typedef struct MLIR_Region MLIR_Region;
typedef struct MLIR_Block MLIR_Block;
typedef struct MLIR_Value MLIR_Value;
typedef struct MLIR_Type MLIR_Type;
typedef struct MLIR_Attribute MLIR_Attribute;
typedef struct MLIR_Location MLIR_Location;

typedef struct MLIR_LocationMap {
    void *impl;
} MLIR_LocationMap;

size_t MLIR_GetLocationMapSize(const MLIR_LocationMap *location_map);
size_t MLIR_CollectLocationMap(const MLIR_LocationMap *location_map, string *out_keys, MLIR_Location **out_locs, size_t max);

// Vector helpers used by the parser implementation
DEFINE_VECTOR_FOR_TYPE(MLIR_Op*, VecOp)
DEFINE_VECTOR_FOR_TYPE(MLIR_Value*, VecValue)
DEFINE_VECTOR_FOR_TYPE(MLIR_Block*, VecBlock)
DEFINE_VECTOR_FOR_TYPE(MLIR_Attribute*, VecAttribute)

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

void MLIR_InitApi(MLIR_Op *root);
void MLIR_SetGlobalArenaAllocator(Arena *arena);

// -----------------------------------------------------------------------------
// Operation API
// -----------------------------------------------------------------------------

// Creation & structural mutation
MLIR_Op *MLIR_CreateOp(
    MLIR_OpType type,
    string opname,
    MLIR_Attribute **attributes, size_t n_attributes,
    MLIR_Type **result_types, size_t n_result_types,
    MLIR_Value **results, size_t n_results,
    MLIR_Value **operands, size_t n_operands,
    MLIR_Region **regions, size_t n_regions,
    MLIR_Location *location,
    MLIR_Location *unnumbered_loc_def,
    string trailing_comment,
    int64_t source_line_start);
void MLIR_AppendOpAttribute(MLIR_Op *op, MLIR_Attribute *attr);

// Accessors
MLIR_OpType MLIR_GetOpType(const MLIR_Op *op);
string MLIR_GetOpName(const MLIR_Op *op);
string MLIR_GetOpName_string(const MLIR_Op *op);
MLIR_Location *MLIR_GetOpLocation(const MLIR_Op *op);
string MLIR_GetOpTrailingComment(const MLIR_Op *op);
int64_t MLIR_GetOpSourceLineStart(const MLIR_Op *op);
MLIR_Location *MLIR_GetOpUnnumberedLocationDef(const MLIR_Op *op);
size_t MLIR_GetOpNumOperands(const MLIR_Op *op);
MLIR_Value *MLIR_GetOpOperand(const MLIR_Op *op, size_t idx);
size_t MLIR_GetOpNumResults(const MLIR_Op *op);
MLIR_Value *MLIR_GetOpResult(const MLIR_Op *op, size_t idx);
size_t MLIR_GetOpNumResultTypes(const MLIR_Op *op);
MLIR_Type *MLIR_GetOpResult_type(const MLIR_Op *op, size_t idx);
size_t MLIR_GetOpNumAttributes(const MLIR_Op *op);
MLIR_Attribute *MLIR_GetOpAttribute(const MLIR_Op *op, size_t idx);
size_t MLIR_GetOpNumRegions(const MLIR_Op *op);
MLIR_Region *MLIR_GetOpRegion(const MLIR_Op *op, size_t idx);

// -----------------------------------------------------------------------------
// Region API
// -----------------------------------------------------------------------------

MLIR_Region *MLIR_CreateRegion(void);
void MLIR_AppendRegionBlock(MLIR_Region *region, MLIR_Block *block);
size_t MLIR_GetRegionNumBlocks(const MLIR_Region *region);
MLIR_Block *MLIR_GetRegionBlock(const MLIR_Region *region, size_t idx);

// -----------------------------------------------------------------------------
// Block API
// -----------------------------------------------------------------------------

MLIR_Block *MLIR_CreateBlock(void);
void MLIR_AppendBlockOp(MLIR_Block *block, MLIR_Op *op);
void MLIR_AppendBlockArg(MLIR_Block *block, MLIR_Value *arg);
size_t MLIR_GetBlockNumOps(const MLIR_Block *block);
MLIR_Op *MLIR_GetBlockOp(const MLIR_Block *block, size_t idx);
size_t MLIR_GetBlockNumArgs(const MLIR_Block *block);
MLIR_Value *MLIR_GetBlockArg(const MLIR_Block *block, size_t idx);

// -----------------------------------------------------------------------------
// Value API
// -----------------------------------------------------------------------------

// Value kind used for SSA values
typedef enum MLIR_ValueKind {
    BLOCK_ARG,
    OP_RESULT
} MLIR_ValueKind;

// Creation
MLIR_Value *MLIR_CreateValueBlockArg(string register_name, uint32_t result_index, MLIR_Type *type, MLIR_Location *location);
MLIR_Value *MLIR_CreateValueOpResult(void *def, uint32_t result_index, MLIR_Type *type, string register_name, MLIR_Location *location);

// Accessors
MLIR_ValueKind MLIR_GetValueKind(const MLIR_Value *value);
MLIR_Type *MLIR_GetValueType(const MLIR_Value *value);
string MLIR_GetValueRegisterName(const MLIR_Value *value);
uint32_t MLIR_GetValueResultIndex(const MLIR_Value *value);
MLIR_Op *MLIR_GetValueDefiningOp(const MLIR_Value *value);
MLIR_Location *MLIR_GetValueLocation(const MLIR_Value *value);

// -----------------------------------------------------------------------------
// Type API
// -----------------------------------------------------------------------------

// Creation
MLIR_Type *MLIR_CreateTypeInteger(uint32_t width, bool is_signed);
MLIR_Type *MLIR_CreateTypeFloat(uint32_t width, bool is_bfloat);
MLIR_Type *MLIR_CreateTypeIndex(void);
MLIR_Type *MLIR_CreateTypeUnknown(void);
MLIR_Type *MLIR_CreateTypeTensor(const int64_t *shape, size_t rank, MLIR_Type *element_type);
MLIR_Type *MLIR_CreateTypeMemref(const int64_t *shape, size_t rank, MLIR_Type *element_type);
MLIR_Type *MLIR_CreateTypePointer(MLIR_Type *element_type, bool has_address_space, uint32_t address_space);
MLIR_Type *MLIR_CreateTypeOpaque(string name);

// Mutation
void MLIR_SetTypeIntegerProperties(MLIR_Type *type, uint32_t width, bool is_signed);
void MLIR_SetTypeFloatProperties(MLIR_Type *type, uint32_t width, bool is_bfloat);
void MLIR_SetTypeTensorProperties(MLIR_Type *type, const int64_t *shape, size_t rank, MLIR_Type *element_type);
void MLIR_SetTypeMemrefProperties(MLIR_Type *type, const int64_t *shape, size_t rank, MLIR_Type *element_type);
void MLIR_SetTypePointerProperties(MLIR_Type *type, MLIR_Type *element_type, bool has_address_space, uint32_t address_space);

// Introspection & formatting
bool MLIR_IsTypeInteger(const MLIR_Type *type);
bool MLIR_IsTypeFloat(const MLIR_Type *type);
bool MLIR_IsTypeTensor(const MLIR_Type *type);
bool MLIR_IsTypeMemref(const MLIR_Type *type);
bool MLIR_IsTypePointer(const MLIR_Type *type);
bool MLIR_IsTypeIndex(const MLIR_Type *type);
bool MLIR_IsTypeUnknown(const MLIR_Type *type);
bool MLIR_IsTypeOpaque(const MLIR_Type *type);
string MLIR_GetTypeString(Arena *arena, MLIR_Type *type);

// -----------------------------------------------------------------------------
// Attribute API
// -----------------------------------------------------------------------------

// Creation & mutation
MLIR_Attribute *MLIR_CreateAttributeInteger(string name, int64_t value);
MLIR_Attribute *MLIR_CreateAttributeFloat(string name, double value);
MLIR_Attribute *MLIR_CreateAttributeBool(string name, bool value);
MLIR_Attribute *MLIR_CreateAttributeString(string name, string value);
MLIR_Attribute *MLIR_CreateAttributeArray(string name, MLIR_Attribute **elements, size_t count);
MLIR_Attribute *MLIR_CreateAttributeDict(string name, MLIR_Attribute **elements, size_t count);

// Introspection
typedef enum {
    MLIR_ATTR_KIND_INTEGER,
    MLIR_ATTR_KIND_FLOAT,
    MLIR_ATTR_KIND_STRING,
    MLIR_ATTR_KIND_BOOL,
    MLIR_ATTR_KIND_ARRAY,
    MLIR_ATTR_KIND_DICT
} MLIR_AttrKind;

MLIR_AttrKind MLIR_GetAttributeKind(const MLIR_Attribute *attr);
string MLIR_GetAttributeName(const MLIR_Attribute *attr);
int64_t MLIR_GetAttributeInteger(const MLIR_Attribute *attr);
double MLIR_GetAttributeFloat(const MLIR_Attribute *attr);
bool MLIR_GetAttributeBool(const MLIR_Attribute *attr);
string MLIR_GetAttributeString(const MLIR_Attribute *attr);
size_t MLIR_GetAttributeArraySize(const MLIR_Attribute *attr);
MLIR_Attribute *MLIR_GetAttributeArrayElement(const MLIR_Attribute *attr, size_t idx);
size_t MLIR_GetAttributeDictSize(const MLIR_Attribute *attr);
MLIR_Attribute *MLIR_GetAttributeDictElement(const MLIR_Attribute *attr, size_t idx);

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
MLIR_Location *MLIR_CreateLocationUnknown(string original_text);
MLIR_Location *MLIR_CreateLocationFile(string filename, int line, int column);
MLIR_Location *MLIR_CreateLocationName(string name);
MLIR_Location *MLIR_CreateLocationRef(int ref_id);

// Accessors
MLIR_LocationKind MLIR_GetLocationKind(const MLIR_Location *loc);
string MLIR_GetLocationOriginalText(const MLIR_Location *loc);
string MLIR_GetLocationFileFilename(const MLIR_Location *loc);
int MLIR_GetLocationFileLine(const MLIR_Location *loc);
int MLIR_GetLocationFileColumn(const MLIR_Location *loc);
string MLIR_GetLocationName(const MLIR_Location *loc);
int MLIR_GetLocationRefId(const MLIR_Location *loc);

#ifdef __cplusplus
}
#endif

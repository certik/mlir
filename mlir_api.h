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

size_t MLIR_LocationMapSize(const MLIR_LocationMap *location_map);
size_t MLIR_LocationMapCollect(const MLIR_LocationMap *location_map, string *out_keys, MLIR_Location **out_locs, size_t max);

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

void MLIR_ApiInit(MLIR_Op *root);

// -----------------------------------------------------------------------------
// Operation API
// -----------------------------------------------------------------------------

// Creation & structural mutation
MLIR_Op *MLIR_OpCreate(
    Arena *arena,
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
void MLIR_OpAppendAttribute(Arena *arena, MLIR_Op *op, MLIR_Attribute *attr);

// Accessors
MLIR_OpType MLIR_OpGetType(const MLIR_Op *op);
string MLIR_OpGetName(const MLIR_Op *op);
string MLIR_OpGetName_string(const MLIR_Op *op);
MLIR_Location *MLIR_OpGetLocation(const MLIR_Op *op);
string MLIR_OpGetTrailingComment(const MLIR_Op *op);
int64_t MLIR_OpGetSourceLineStart(const MLIR_Op *op);
MLIR_Location *MLIR_OpGetUnnumberedLocDef(const MLIR_Op *op);
size_t MLIR_OpNumOperands(const MLIR_Op *op);
MLIR_Value *MLIR_OpGetOperand(const MLIR_Op *op, size_t idx);
size_t MLIR_OpNumResults(const MLIR_Op *op);
MLIR_Value *MLIR_OpGetResult(const MLIR_Op *op, size_t idx);
size_t MLIR_OpNumResultTypes(const MLIR_Op *op);
MLIR_Type *MLIR_OpGetResult_type(const MLIR_Op *op, size_t idx);
size_t MLIR_OpNumAttributes(const MLIR_Op *op);
MLIR_Attribute *MLIR_OpGetAttribute(const MLIR_Op *op, size_t idx);
size_t MLIR_OpNumRegions(const MLIR_Op *op);
MLIR_Region *MLIR_OpGetRegion(const MLIR_Op *op, size_t idx);

// -----------------------------------------------------------------------------
// Region API
// -----------------------------------------------------------------------------

MLIR_Region *MLIR_RegionCreate(Arena *arena);
void MLIR_RegionAddBlock(Arena *arena, MLIR_Region *region, MLIR_Block *block);
size_t MLIR_RegionNumBlocks(const MLIR_Region *region);
MLIR_Block *MLIR_RegionGetBlock(const MLIR_Region *region, size_t idx);

// -----------------------------------------------------------------------------
// Block API
// -----------------------------------------------------------------------------

MLIR_Block *MLIR_BlockCreate(Arena *arena);
void MLIR_BlockAddOp(Arena *arena, MLIR_Block *block, MLIR_Op *op);
void MLIR_BlockAddArg(Arena *arena, MLIR_Block *block, MLIR_Value *arg);
size_t MLIR_BlockNumOps(const MLIR_Block *block);
MLIR_Op *MLIR_BlockGetOp(const MLIR_Block *block, size_t idx);
size_t MLIR_BlockNumArgs(const MLIR_Block *block);
MLIR_Value *MLIR_BlockGetArg(const MLIR_Block *block, size_t idx);

// -----------------------------------------------------------------------------
// Value API
// -----------------------------------------------------------------------------

// Value kind used for SSA values
typedef enum MLIR_ValueKind {
    BLOCK_ARG,
    OP_RESULT
} MLIR_ValueKind;

// Creation
MLIR_Value *MLIR_ValueCreateBlockArg(Arena *arena, string register_name, uint32_t result_index, MLIR_Type *type, MLIR_Location *location);
MLIR_Value *MLIR_ValueCreateOpResult(Arena *arena, void *def, uint32_t result_index, MLIR_Type *type, string register_name, MLIR_Location *location);

// Accessors
MLIR_ValueKind MLIR_ValueGetKind(const MLIR_Value *value);
MLIR_Type *MLIR_ValueGetType(const MLIR_Value *value);
string MLIR_ValueGetRegisterName(const MLIR_Value *value);
uint32_t MLIR_ValueGetResultIndex(const MLIR_Value *value);
MLIR_Op *MLIR_ValueGetDefOp(const MLIR_Value *value);
MLIR_Location *MLIR_ValueGetLocation(const MLIR_Value *value);

// -----------------------------------------------------------------------------
// Type API
// -----------------------------------------------------------------------------

// Creation
MLIR_Type *MLIR_TypeCreateInteger(Arena *arena, uint32_t width, bool is_signed);
MLIR_Type *MLIR_TypeCreateFloat(Arena *arena, uint32_t width, bool is_bfloat);
MLIR_Type *MLIR_TypeCreateIndex(Arena *arena);
MLIR_Type *MLIR_TypeCreateUnknown(Arena *arena);
MLIR_Type *MLIR_TypeCreateTensor(Arena *arena, const int64_t *shape, size_t rank, MLIR_Type *element_type);
MLIR_Type *MLIR_TypeCreateMemref(Arena *arena, const int64_t *shape, size_t rank, MLIR_Type *element_type);
MLIR_Type *MLIR_TypeCreatePointer(Arena *arena, MLIR_Type *element_type, bool has_address_space, uint32_t address_space);
MLIR_Type *MLIR_TypeCreateOpaque(Arena *arena, string name);

// Mutation
void MLIR_TypeSetIntegerProperties(MLIR_Type *type, uint32_t width, bool is_signed);
void MLIR_TypeSetFloatProperties(MLIR_Type *type, uint32_t width, bool is_bfloat);
void MLIR_TypeSetTensorProperties(MLIR_Type *type, const int64_t *shape, size_t rank, MLIR_Type *element_type);
void MLIR_TypeSetMemrefProperties(MLIR_Type *type, const int64_t *shape, size_t rank, MLIR_Type *element_type);
void MLIR_TypeSetPointerProperties(MLIR_Type *type, MLIR_Type *element_type, bool has_address_space, uint32_t address_space);

// Introspection & formatting
bool MLIR_TypeIsInteger(const MLIR_Type *type);
bool MLIR_TypeIsFloat(const MLIR_Type *type);
bool MLIR_TypeIsTensor(const MLIR_Type *type);
bool MLIR_TypeIsMemref(const MLIR_Type *type);
bool MLIR_TypeIsPointer(const MLIR_Type *type);
bool MLIR_TypeIsIndex(const MLIR_Type *type);
bool MLIR_TypeIsUnknown(const MLIR_Type *type);
bool MLIR_TypeIsOpaque(const MLIR_Type *type);
string MLIR_TypeToString(Arena *arena, MLIR_Type *type);

// -----------------------------------------------------------------------------
// Attribute API
// -----------------------------------------------------------------------------

// Creation & mutation
MLIR_Attribute *MLIR_AttributeCreateInteger(Arena *arena, string name, int64_t value);
MLIR_Attribute *MLIR_AttributeCreateFloat(Arena *arena, string name, double value);
MLIR_Attribute *MLIR_AttributeCreateBool(Arena *arena, string name, bool value);
MLIR_Attribute *MLIR_AttributeCreateString(Arena *arena, string name, string value);
MLIR_Attribute *MLIR_AttributeCreateArray(Arena *arena, string name, MLIR_Attribute **elements, size_t count);
MLIR_Attribute *MLIR_AttributeCreateDict(Arena *arena, string name, MLIR_Attribute **elements, size_t count);

// Introspection
typedef enum {
    MLIR_ATTR_KIND_INTEGER,
    MLIR_ATTR_KIND_FLOAT,
    MLIR_ATTR_KIND_STRING,
    MLIR_ATTR_KIND_BOOL,
    MLIR_ATTR_KIND_ARRAY,
    MLIR_ATTR_KIND_DICT
} MLIR_AttrKind;

MLIR_AttrKind MLIR_AttributeGetKind(const MLIR_Attribute *attr);
string MLIR_AttributeGetName(const MLIR_Attribute *attr);
int64_t MLIR_AttributeGetInteger(const MLIR_Attribute *attr);
double MLIR_AttributeGetFloat(const MLIR_Attribute *attr);
bool MLIR_AttributeGetBool(const MLIR_Attribute *attr);
string MLIR_AttributeGetString(const MLIR_Attribute *attr);
size_t MLIR_AttributeGetArraySize(const MLIR_Attribute *attr);
MLIR_Attribute *MLIR_AttributeGetArrayElement(const MLIR_Attribute *attr, size_t idx);
size_t MLIR_AttributeGetDictSize(const MLIR_Attribute *attr);
MLIR_Attribute *MLIR_AttributeGetDictElement(const MLIR_Attribute *attr, size_t idx);

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
MLIR_Location *MLIR_LocationCreateUnknown(Arena *arena, string original_text);
MLIR_Location *MLIR_LocationCreateFile(Arena *arena, string filename, int line, int column);
MLIR_Location *MLIR_LocationCreateName(Arena *arena, string name);
MLIR_Location *MLIR_LocationCreateRef(Arena *arena, int ref_id);

// Accessors
MLIR_LocationKind MLIR_LocationGetKind(const MLIR_Location *loc);
string MLIR_LocationGetOriginalText(const MLIR_Location *loc);
string MLIR_LocationGetFileFilename(const MLIR_Location *loc);
int MLIR_LocationGetFileLine(const MLIR_Location *loc);
int MLIR_LocationGetFileColumn(const MLIR_Location *loc);
string MLIR_LocationGetName(const MLIR_Location *loc);
int MLIR_LocationGetRefId(const MLIR_Location *loc);

#ifdef __cplusplus
}
#endif

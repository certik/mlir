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

typedef struct MlirOperation MlirOperation;
typedef struct MlirRegion MlirRegion;
typedef struct MlirBlock MlirBlock;
typedef struct MlirValue MlirValue;
typedef struct MlirType MlirType;
typedef struct MlirAttribute MlirAttribute;
typedef struct MlirLocation MlirLocation;

typedef struct MlirLocationMap {
    void *impl;
} MlirLocationMap;

size_t MLIR_LocationMapSize(const MlirLocationMap *location_map);
size_t MLIR_LocationMapCollect(const MlirLocationMap *location_map, string *out_keys, MlirLocation **out_locs, size_t max);

// Vector helpers used by the parser implementation
DEFINE_VECTOR_FOR_TYPE(MlirOperation*, VecOperation)
DEFINE_VECTOR_FOR_TYPE(MlirValue*, VecValue)
DEFINE_VECTOR_FOR_TYPE(MlirBlock*, VecBlock)
DEFINE_VECTOR_FOR_TYPE(MlirAttribute*, VecAttribute)

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
} OpType;

string MLIR_OpTypeToString(OpType type);

// -----------------------------------------------------------------------------
// API lifecycle
// -----------------------------------------------------------------------------

void MLIR_ApiInit(MlirOperation *root);

// -----------------------------------------------------------------------------
// Operation API
// -----------------------------------------------------------------------------

// Creation & structural mutation
MlirOperation *MLIR_OpCreate(
    Arena *arena,
    OpType type,
    string opname,
    MlirAttribute **attributes, size_t n_attributes,
    MlirType **result_types, size_t n_result_types,
    MlirValue **results, size_t n_results,
    MlirValue **operands, size_t n_operands,
    MlirRegion **regions, size_t n_regions,
    MlirLocation *location,
    MlirLocation *unnumbered_loc_def,
    string trailing_comment,
    int64_t source_line_start);
void MLIR_OpAppendAttribute(Arena *arena, MlirOperation *op, MlirAttribute *attr);

// Accessors
OpType MLIR_OpGetType(const MlirOperation *op);
string MLIR_OpGetName(const MlirOperation *op);
string MLIR_OpGetName_string(const MlirOperation *op);
MlirLocation *MLIR_OpGetLocation(const MlirOperation *op);
string MLIR_OpGetTrailingComment(const MlirOperation *op);
int64_t MLIR_OpGetSourceLineStart(const MlirOperation *op);
MlirLocation *MLIR_OpGetUnnumberedLocDef(const MlirOperation *op);
size_t MLIR_OpNumOperands(const MlirOperation *op);
MlirValue *MLIR_OpGetOperand(const MlirOperation *op, size_t idx);
size_t MLIR_OpNumResults(const MlirOperation *op);
MlirValue *MLIR_OpGetResult(const MlirOperation *op, size_t idx);
size_t MLIR_OpNumResultTypes(const MlirOperation *op);
MlirType *MLIR_OpGetResult_type(const MlirOperation *op, size_t idx);
size_t MLIR_OpNumAttributes(const MlirOperation *op);
MlirAttribute *MLIR_OpGetAttribute(const MlirOperation *op, size_t idx);
size_t MLIR_OpNumRegions(const MlirOperation *op);
MlirRegion *MLIR_OpGetRegion(const MlirOperation *op, size_t idx);

// -----------------------------------------------------------------------------
// Region API
// -----------------------------------------------------------------------------

MlirRegion *MLIR_RegionCreate(Arena *arena);
void MLIR_RegionAddBlock(Arena *arena, MlirRegion *region, MlirBlock *block);
size_t MLIR_RegionNumBlocks(const MlirRegion *region);
MlirBlock *MLIR_RegionGetBlock(const MlirRegion *region, size_t idx);

// -----------------------------------------------------------------------------
// Block API
// -----------------------------------------------------------------------------

MlirBlock *MLIR_BlockCreate(Arena *arena);
void MLIR_BlockAddOp(Arena *arena, MlirBlock *block, MlirOperation *op);
void MLIR_BlockAddArg(Arena *arena, MlirBlock *block, MlirValue *arg);
size_t MLIR_BlockNumOps(const MlirBlock *block);
MlirOperation *MLIR_BlockGetOp(const MlirBlock *block, size_t idx);
size_t MLIR_BlockNumArgs(const MlirBlock *block);
MlirValue *MLIR_BlockGetArg(const MlirBlock *block, size_t idx);

// -----------------------------------------------------------------------------
// Value API
// -----------------------------------------------------------------------------

// Value kind used for SSA values
typedef enum ValueKind {
    BLOCK_ARG,
    OP_RESULT
} ValueKind;

// Creation
MlirValue *MLIR_ValueCreateBlockArg(Arena *arena, string register_name, uint32_t result_index, MlirType *type, MlirLocation *location);
MlirValue *MLIR_ValueCreateOpResult(Arena *arena, void *def, uint32_t result_index, MlirType *type, string register_name, MlirLocation *location);

// Accessors
ValueKind MLIR_ValueGetKind(const MlirValue *value);
MlirType *MLIR_ValueGetType(const MlirValue *value);
string MLIR_ValueGetRegisterName(const MlirValue *value);
uint32_t MLIR_ValueGetResultIndex(const MlirValue *value);
MlirOperation *MLIR_ValueGetDefOp(const MlirValue *value);
MlirLocation *MLIR_ValueGetLocation(const MlirValue *value);

// -----------------------------------------------------------------------------
// Type API
// -----------------------------------------------------------------------------

// Creation
MlirType *MLIR_TypeCreateInteger(Arena *arena, uint32_t width, bool is_signed);
MlirType *MLIR_TypeCreateFloat(Arena *arena, uint32_t width, bool is_bfloat);
MlirType *MLIR_TypeCreateIndex(Arena *arena);
MlirType *MLIR_TypeCreateUnknown(Arena *arena);
MlirType *MLIR_TypeCreateTensor(Arena *arena, const int64_t *shape, size_t rank, MlirType *element_type);
MlirType *MLIR_TypeCreateMemref(Arena *arena, const int64_t *shape, size_t rank, MlirType *element_type);
MlirType *MLIR_TypeCreatePointer(Arena *arena, MlirType *element_type, bool has_address_space, uint32_t address_space);
MlirType *MLIR_TypeCreateOpaque(Arena *arena, string name);

// Mutation
void MLIR_TypeSetIntegerProperties(MlirType *type, uint32_t width, bool is_signed);
void MLIR_TypeSetFloatProperties(MlirType *type, uint32_t width, bool is_bfloat);
void MLIR_TypeSetTensorProperties(MlirType *type, const int64_t *shape, size_t rank, MlirType *element_type);
void MLIR_TypeSetMemrefProperties(MlirType *type, const int64_t *shape, size_t rank, MlirType *element_type);
void MLIR_TypeSetPointerProperties(MlirType *type, MlirType *element_type, bool has_address_space, uint32_t address_space);

// Introspection & formatting
bool MLIR_TypeIsInteger(const MlirType *type);
bool MLIR_TypeIsFloat(const MlirType *type);
bool MLIR_TypeIsTensor(const MlirType *type);
bool MLIR_TypeIsMemref(const MlirType *type);
bool MLIR_TypeIsPointer(const MlirType *type);
bool MLIR_TypeIsIndex(const MlirType *type);
bool MLIR_TypeIsUnknown(const MlirType *type);
bool MLIR_TypeIsOpaque(const MlirType *type);
string MLIR_TypeToString(Arena *arena, MlirType *type);

// -----------------------------------------------------------------------------
// Attribute API
// -----------------------------------------------------------------------------

// Creation & mutation
MlirAttribute *MLIR_AttributeCreateInteger(Arena *arena, string name, int64_t value);
MlirAttribute *MLIR_AttributeCreateFloat(Arena *arena, string name, double value);
MlirAttribute *MLIR_AttributeCreateBool(Arena *arena, string name, bool value);
MlirAttribute *MLIR_AttributeCreateString(Arena *arena, string name, string value);
MlirAttribute *MLIR_AttributeCreateArray(Arena *arena, string name, MlirAttribute **elements, size_t count);
MlirAttribute *MLIR_AttributeCreateDict(Arena *arena, string name, MlirAttribute **elements, size_t count);

// Introspection
typedef enum {
    MLIR_ATTR_KIND_INTEGER,
    MLIR_ATTR_KIND_FLOAT,
    MLIR_ATTR_KIND_STRING,
    MLIR_ATTR_KIND_BOOL,
    MLIR_ATTR_KIND_ARRAY,
    MLIR_ATTR_KIND_DICT
} MlirAttrKind;

MlirAttrKind MLIR_AttributeGetKind(const MlirAttribute *attr);
string MLIR_AttributeGetName(const MlirAttribute *attr);
int64_t MLIR_AttributeGetInteger(const MlirAttribute *attr);
double MLIR_AttributeGetFloat(const MlirAttribute *attr);
bool MLIR_AttributeGetBool(const MlirAttribute *attr);
string MLIR_AttributeGetString(const MlirAttribute *attr);
size_t MLIR_AttributeGetArraySize(const MlirAttribute *attr);
MlirAttribute *MLIR_AttributeGetArrayElement(const MlirAttribute *attr, size_t idx);
size_t MLIR_AttributeGetDictSize(const MlirAttribute *attr);
MlirAttribute *MLIR_AttributeGetDictElement(const MlirAttribute *attr, size_t idx);

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
} MlirLocationKind;

// Creation
MlirLocation *MLIR_LocationCreateUnknown(Arena *arena, string original_text);
MlirLocation *MLIR_LocationCreateFile(Arena *arena, string filename, int line, int column);
MlirLocation *MLIR_LocationCreateName(Arena *arena, string name);
MlirLocation *MLIR_LocationCreateRef(Arena *arena, int ref_id);

// Accessors
MlirLocationKind MLIR_LocationGetKind(const MlirLocation *loc);
string MLIR_LocationGetOriginalText(const MlirLocation *loc);
string MLIR_LocationGetFileFilename(const MlirLocation *loc);
int MLIR_LocationGetFileLine(const MlirLocation *loc);
int MLIR_LocationGetFileColumn(const MlirLocation *loc);
string MLIR_LocationGetName(const MlirLocation *loc);
int MLIR_LocationGetRefId(const MlirLocation *loc);

#ifdef __cplusplus
}
#endif

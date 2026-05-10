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
    OP_TYPE_ARITH_FPTOSI,
    OP_TYPE_ARITH_INDEX_CAST,
    OP_TYPE_ARITH_EXTSI,
    OP_TYPE_ARITH_TRUNCI,
    OP_TYPE_ARITH_EXTF,
    OP_TYPE_ARITH_TRUNCF,
    OP_TYPE_ARITH_EXTUI,
    OP_TYPE_ARITH_MAXF,
    OP_TYPE_ARITH_DIVSI,
    OP_TYPE_ARITH_REMSI,
    OP_TYPE_ARITH_DIVUI,
    OP_TYPE_ARITH_REMUI,
    OP_TYPE_ARITH_SHRUI,
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
    OP_TYPE_FUNC_CALL_INDIRECT,
    OP_TYPE_FUNC_CONSTANT,
    OP_TYPE_UNREALIZED_CONVERSION_CAST,

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
    OP_TYPE_LINALG_COPY,

    // Index dialect
    OP_TYPE_INDEX_CONSTANT,

    // LLVM dialect
    OP_TYPE_LLVM_MLIR_UNDEF,
    OP_TYPE_LLVM_ALLOCA,
    OP_TYPE_LLVM_LOAD,
    OP_TYPE_LLVM_STORE,
    OP_TYPE_LLVM_GEP,
    OP_TYPE_LLVM_MLIR_ZERO,
    OP_TYPE_LLVM_MLIR_CONSTANT,
    OP_TYPE_LLVM_ICMP,
    OP_TYPE_LLVM_MLIR_ADDRESSOF,
    OP_TYPE_LLVM_MLIR_GLOBAL,
    OP_TYPE_LLVM_RETURN,
    OP_TYPE_LLVM_PTRTOINT,
    OP_TYPE_LLVM_FUNC,
    OP_TYPE_LLVM_CALL,
    OP_TYPE_LLVM_SEXT,
    OP_TYPE_LLVM_ADD,
    OP_TYPE_LLVM_SUB,
    OP_TYPE_LLVM_MUL,
    OP_TYPE_LLVM_SDIV,
    OP_TYPE_LLVM_UDIV,
    OP_TYPE_LLVM_SREM,
    OP_TYPE_LLVM_UREM,
    OP_TYPE_LLVM_AND,
    OP_TYPE_LLVM_OR,
    OP_TYPE_LLVM_XOR,
    OP_TYPE_LLVM_SHL,
    OP_TYPE_LLVM_LSHR,
    OP_TYPE_LLVM_ASHR,
    OP_TYPE_LLVM_TRUNC,
    OP_TYPE_LLVM_ZEXT,
    OP_TYPE_ARITH_XORI,
    OP_TYPE_ARITH_SHLI,
    OP_TYPE_ARITH_SHRSI,

    // Return operations
    OP_TYPE_RETURN,
    OP_TYPE_TT_REDUCE_RETURN,

    // -------------------------------------------------------------------------
    // wasmssa dialect — high-level SSA-form WebAssembly ops. Produced by
    // the LLVM-dialect -> wasmssa lowering pass and consumed by the
    // wasmssa -> wasmstack stackification pass. All values are SSA;
    // there are no explicit local.get / local.set / local.tee here.
    //
    // Op operand-order convention (non-commutative ops):
    //   sub:   %r = wasmssa.sub %lhs, %rhs   ; emits "lhs rhs i32.sub"
    //   load:  %r = wasmssa.load %addr       ; offset/align as attrs
    //   store: wasmssa.store %addr, %val
    // -------------------------------------------------------------------------
    OP_TYPE_WASMSSA_FUNC,
    OP_TYPE_WASMSSA_IMPORT_FUNC,
    OP_TYPE_WASMSSA_IMPORT_GLOBAL,
    OP_TYPE_WASMSSA_CONST,
    OP_TYPE_WASMSSA_ADD,
    OP_TYPE_WASMSSA_SUB,
    OP_TYPE_WASMSSA_BINOP,
    OP_TYPE_WASMSSA_UNOP,
    OP_TYPE_WASMSSA_LOAD,
    OP_TYPE_WASMSSA_STORE,
    OP_TYPE_WASMSSA_GLOBAL_GET,
    OP_TYPE_WASMSSA_GLOBAL_SET,
    OP_TYPE_WASMSSA_EXTEND_I32_S,
    OP_TYPE_WASMSSA_RETURN,
    OP_TYPE_WASMSSA_CALL,
    // Structured control-flow markers. Linearized into the flat per-func
    // op stream in stage 1; stage 2 pairs BEGINs to ENDs to compute br
    // depths.
    OP_TYPE_WASMSSA_BLOCK_BEGIN,
    OP_TYPE_WASMSSA_LOOP_BEGIN,
    OP_TYPE_WASMSSA_IF_BEGIN,
    OP_TYPE_WASMSSA_IF_ELSE,
    OP_TYPE_WASMSSA_END,
    OP_TYPE_WASMSSA_BR,
    OP_TYPE_WASMSSA_BR_IF,
    OP_TYPE_WASMSSA_SELECT,
    OP_TYPE_WASMSSA_EQZ,
    // Carrier ops materialize scf.yield-ed values across structured-CF
    // boundaries via per-function shared locals (allocated lazily by
    // stage 2). Each carrier_id refers to one local of a fixed valtype.
    OP_TYPE_WASMSSA_CARRIER_SET,
    OP_TYPE_WASMSSA_CARRIER_GET,

    // -------------------------------------------------------------------------
    // wasmstack dialect — low-level stack-machine WebAssembly ops. 1:1
    // with the wasm bytecode opcodes. Produced by the wasmssa -> wasmstack
    // stackification pass and consumed by the binary emitter.
    // -------------------------------------------------------------------------
    OP_TYPE_WASMSTACK_FUNC,
    OP_TYPE_WASMSTACK_IMPORT_FUNC,
    OP_TYPE_WASMSTACK_IMPORT_GLOBAL,
    OP_TYPE_WASMSTACK_LOCAL_GET,
    OP_TYPE_WASMSTACK_LOCAL_SET,
    OP_TYPE_WASMSTACK_LOCAL_TEE,
    OP_TYPE_WASMSTACK_CONST,
    OP_TYPE_WASMSTACK_ADD,
    OP_TYPE_WASMSTACK_SUB,
    OP_TYPE_WASMSTACK_BINOP,
    OP_TYPE_WASMSTACK_UNOP,
    OP_TYPE_WASMSTACK_LOAD,
    OP_TYPE_WASMSTACK_STORE,
    OP_TYPE_WASMSTACK_GLOBAL_GET,
    OP_TYPE_WASMSTACK_GLOBAL_SET,
    OP_TYPE_WASMSTACK_EXTEND_I32_S,
    OP_TYPE_WASMSTACK_RETURN,
    OP_TYPE_WASMSTACK_CALL,
    // Structured-CF + select.
    OP_TYPE_WASMSTACK_BLOCK,
    OP_TYPE_WASMSTACK_LOOP,
    OP_TYPE_WASMSTACK_IF,
    OP_TYPE_WASMSTACK_ELSE,
    OP_TYPE_WASMSTACK_END,
    OP_TYPE_WASMSTACK_BR,
    OP_TYPE_WASMSTACK_BR_IF,
    OP_TYPE_WASMSTACK_SELECT,
    OP_TYPE_WASMSTACK_EQZ,

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
// Like MLIR_CreateOp, but with explicit successor blocks and per-successor
// operand arrays. Used by branch ops (cf.br, cf.cond_br, ...). The successor
// blocks must be valid block handles (forward-referenced blocks are fine as
// long as MLIR_CreateBlock has been called for them).
MLIR_OpHandle MLIR_CreateOpWithSuccessors(
    MLIR_Context *ctx,
    MLIR_OpType type,
    string opname,
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
    int64_t source_line_start);
void MLIR_AppendOpAttribute(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_AttributeHandle attr);

// Print an operation. Three named entry points instead of an enum-dispatched
// MLIR_PrintOperation: each lives in its own translation unit so the linker
// pulls in only what each binary actually calls. The Upstream variant
// requires the upstream backend (the native backend's stub returns an
// "error: ..." string).
string MLIR_PrintOperationUpstream(MLIR_Context *ctx, MLIR_OpHandle op);
string MLIR_PrintOperationClassic(MLIR_Context *ctx, MLIR_OpHandle op);
string MLIR_PrintOperationGeneric(MLIR_Context *ctx, MLIR_OpHandle op);

// Parse a top-level MLIR module from text. Returns MLIR_INVALID_HANDLE on
// failure (or, for Upstream on the native backend, always).
MLIR_OpHandle MLIR_ParseTextUpstream(MLIR_Context *ctx, string text);
MLIR_OpHandle MLIR_ParseTextClassic(MLIR_Context *ctx, string text);

// Backend selector for the lowering / LLVM-IR translation entry points.
//
// MLIR_LOWERING_UPSTREAM uses upstream MLIR's pass pipeline / translator
// (only available when linked against mlir_api_impl_upstream.cpp).
//
// MLIR_LOWERING_NATIVE uses our own implementation, walking the IR
// through mlir_api.h only. It is the path that makes the native build
// (mlir_api_impl.c) self-contained.
typedef enum {
    MLIR_LOWERING_UPSTREAM = 0,
    MLIR_LOWERING_NATIVE   = 1,
} MLIR_LoweringBackend;

// Lower a `builtin.module` op to the LLVM dialect by running the standard
// conversion passes (scf -> cf, arith/memref/cf/func/vector -> llvm,
// reconcile-unrealized-casts). Mutates `module` in place. Returns true on
// success. With MLIR_LOWERING_UPSTREAM available only when linked against
// the upstream backend; with MLIR_LOWERING_NATIVE this is the native
// implementation. Diagnostic messages from failed passes are printed to
// stderr.
bool MLIR_LowerToLLVMDialect(MLIR_Context *ctx, MLIR_OpHandle module,
                             MLIR_LoweringBackend backend);

// Same as MLIR_LowerToLLVMDialect, but tailored to feed the native
// LLVM-dialect-MLIR -> WASM emitter (mlir_translate_to_wasm.c). The
// difference matters only for MLIR_LOWERING_NATIVE: instead of running
// scf->cf and cf->llvm (which destroys structured control flow that
// wasm needs), this entry point runs upstream's `lift-cf-to-scf`
// pass first and then lowers arith/memref/func/vector/etc. to LLVM,
// leaving scf.* ops in place for stage 1 of the wasm pipeline to
// consume directly. For MLIR_LOWERING_UPSTREAM it is identical to
// MLIR_LowerToLLVMDialect.
bool MLIR_LowerToLLVMDialectForWasm(MLIR_Context *ctx, MLIR_OpHandle module,
                                    MLIR_LoweringBackend backend);

// Translate a `builtin.module` op already lowered to the LLVM dialect into
// LLVM IR text (`.ll`). Returns the IR text on success or an empty string
// on failure. See MLIR_LoweringBackend above for backend semantics.
string MLIR_TranslateModuleToLLVMIR(MLIR_Context *ctx, MLIR_OpHandle module,
                                    MLIR_LoweringBackend backend);

// Translate a `builtin.module` op already lowered to the LLVM dialect into
// a WebAssembly relocatable object file (wasm32-wasi). Returns the raw
// object bytes on success or an empty string on failure. The returned
// bytes are NOT a runnable wasm module — they still need to be linked
// (typically with `wasm-ld`) against any runtime/imports the program
// uses, producing a final `.wasm` module that can be run by wasmtime.
//
// Backend semantics:
//   MLIR_LOWERING_UPSTREAM uses LLVM's WebAssembly target machine to
//   emit the object file. Requires the upstream backend.
//   MLIR_LOWERING_NATIVE is not implemented and returns an empty string
//   (a diagnostic is printed to stderr).
string MLIR_TranslateModuleToWasm(MLIR_Context *ctx, MLIR_OpHandle module,
                                  MLIR_LoweringBackend backend);

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
size_t MLIR_GetOpNumSuccessors(MLIR_OpHandle op);
MLIR_BlockHandle MLIR_GetOpSuccessor(MLIR_OpHandle op, size_t idx);
size_t MLIR_GetOpNumSuccessorOperands(MLIR_OpHandle op, size_t succ_idx);
MLIR_ValueHandle MLIR_GetOpSuccessorOperand(MLIR_OpHandle op, size_t succ_idx, size_t op_idx);

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
void MLIR_InsertBlockOpBeforeTerminator(MLIR_Context *ctx, MLIR_BlockHandle block, MLIR_OpHandle op);
// Insert `op` at position `idx` in `block`'s op list, shifting later ops
// down by one. `idx` is clamped to [0, n_ops].
void MLIR_InsertBlockOpAtIndex(MLIR_Context *ctx, MLIR_BlockHandle block, MLIR_OpHandle op, size_t idx);
void MLIR_AppendBlockArg(MLIR_Context *ctx, MLIR_BlockHandle block, MLIR_ValueHandle arg);
size_t MLIR_GetBlockNumOps(MLIR_BlockHandle block);
MLIR_OpHandle MLIR_GetBlockOp(MLIR_BlockHandle block, size_t idx);
size_t MLIR_GetBlockNumArgs(MLIR_BlockHandle block);
MLIR_ValueHandle MLIR_GetBlockArg(MLIR_BlockHandle block, size_t idx);
// Returns the 0-indexed position of `block` within its parent region.
// Returns SIZE_MAX if the block has no parent or the backend doesn't track it.
size_t MLIR_GetBlockIndex(MLIR_BlockHandle block);

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

// LLVM dialect types — used by frontends that emit `!llvm.ptr`,
// `!llvm.struct`, and `!llvm.array`. Identified structs are mutable: create
// with MLIR_CreateTypeLLVMStructIdentified, then call
// MLIR_SetTypeLLVMStructBody once. Recursion through `!llvm.ptr` (which is
// opaque) is fully supported because the body never mentions the recursive
// struct's body type.
MLIR_TypeHandle MLIR_CreateTypeLLVMPointer(MLIR_Context *ctx);
MLIR_TypeHandle MLIR_CreateTypeLLVMStructIdentified(MLIR_Context *ctx, string name);
void            MLIR_SetTypeLLVMStructBody(MLIR_Context *ctx, MLIR_TypeHandle struct_ty,
                                           const MLIR_TypeHandle *fields, size_t n_fields);
MLIR_TypeHandle MLIR_CreateTypeLLVMArray(MLIR_Context *ctx, MLIR_TypeHandle elem, uint64_t count);

// LLVM-dialect global helpers. Each returns a freshly-created (unattached)
// op; the caller appends it to the module body. Implemented by the
// upstream backend; the native backend returns MLIR_INVALID_HANDLE.
//
// MLIR_CreateLLVMGlobalString: emits
//   llvm.mlir.global private constant @<sym>("<bytes>") : !llvm.array<N x i8>
//
// MLIR_CreateLLVMGlobal: emits
//   llvm.mlir.global internal @<sym>(<init>) : <elem_ty>
// Init shapes:
//   init_kind == 0  -> simple integer initializer (init_int)
//   init_kind == 1  -> simple float initializer (init_float)
//   init_kind == 2  -> region initializer: caller emits ops into
//                      *out_init_block and terminates with llvm.return.
//   init_kind == 3  -> no initializer (zero-initialized).
//   init_kind == 4  -> external linkage, no initializer (for `extern T x;`
//                      declarations resolved by the linker, e.g. `stdout`).
MLIR_OpHandle MLIR_CreateLLVMGlobalString(MLIR_Context *ctx,
                                          string sym_name,
                                          string bytes,
                                          MLIR_LocationHandle loc);
MLIR_OpHandle MLIR_CreateLLVMGlobal(MLIR_Context *ctx,
                                    string sym_name,
                                    MLIR_TypeHandle elem_ty,
                                    bool is_constant,
                                    int init_kind,
                                    int64_t init_int,
                                    double init_float,
                                    MLIR_BlockHandle *out_init_block,
                                    MLIR_LocationHandle loc);

MLIR_TypeHandle MLIR_CreateTypeOpaque(MLIR_Context *ctx, string name);
// Function type: (input_types) -> (result_types). Both arrays are
// copied; passing 0 inputs/results is allowed.
MLIR_TypeHandle MLIR_CreateTypeFunction(MLIR_Context *ctx,
                                         const MLIR_TypeHandle *inputs, size_t n_inputs,
                                         const MLIR_TypeHandle *results, size_t n_results);

// LLVM-dialect function type (LLVMFunctionType). Differs from the standard
// FunctionType in two ways: it has at most one result (use the LLVM `void`
// type for "no result"), and it has an `is_var_arg` flag used to model C
// variadic functions like `int printf(const char *, ...)`. The result type
// must be either an LLVM-compatible scalar/pointer type or the LLVM void
// type (see MLIR_CreateTypeLLVMVoid).
MLIR_TypeHandle MLIR_CreateTypeLLVMFunction(MLIR_Context *ctx,
                                             MLIR_TypeHandle result,
                                             const MLIR_TypeHandle *inputs,
                                             size_t n_inputs,
                                             bool is_var_arg);
// LLVM `void` type — only valid as the result of an LLVMFunctionType.
MLIR_TypeHandle MLIR_CreateTypeLLVMVoid(MLIR_Context *ctx);

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
bool MLIR_IsTypeFunction(MLIR_TypeHandle type);
size_t MLIR_GetTypeFunctionNumInputs(MLIR_TypeHandle type);
MLIR_TypeHandle MLIR_GetTypeFunctionInput(MLIR_TypeHandle type, size_t idx);
size_t MLIR_GetTypeFunctionNumResults(MLIR_TypeHandle type);
MLIR_TypeHandle MLIR_GetTypeFunctionResult(MLIR_TypeHandle type, size_t idx);
// True if the type is an `LLVMFunctionType` whose `isVarArg` flag is set.
// Returns false for the standard `FunctionType` (which is never variadic).
bool MLIR_GetTypeFunctionIsVarArg(MLIR_TypeHandle type);
// Element type of a tensor/memref/vector type, or invalid if the type is
// not a shaped type.
MLIR_TypeHandle MLIR_GetTypeShapedElement(MLIR_TypeHandle type);
string MLIR_GetTypeString(MLIR_Context *ctx, MLIR_TypeHandle type);

// -----------------------------------------------------------------------------
// Attribute API
// -----------------------------------------------------------------------------

// Creation & mutation
//
// Integer and Float attributes carry an MLIR Type indicating their numeric
// type (e.g. i32, i64, index, f32, f64). The caller must construct that
// type first via the Type API. There is no untyped form: typing every
// numeric attribute eagerly is what allows ops like arith.constant to
// build proper IR (where `value`'s attribute type must equal the result
// type) without later fixups.
MLIR_AttributeHandle MLIR_CreateAttributeInteger(MLIR_Context *ctx, string name, int64_t value, MLIR_TypeHandle type);
MLIR_AttributeHandle MLIR_CreateAttributeFloat(MLIR_Context *ctx, string name, double value, MLIR_TypeHandle type);
MLIR_AttributeHandle MLIR_CreateAttributeBool(MLIR_Context *ctx, string name, bool value);
MLIR_AttributeHandle MLIR_CreateAttributeString(MLIR_Context *ctx, string name, string value);
MLIR_AttributeHandle MLIR_CreateAttributeLLVMLinkageInternal(MLIR_Context *ctx, string name);
MLIR_AttributeHandle MLIR_CreateAttributeArray(MLIR_Context *ctx, string name, MLIR_AttributeHandle *elements, size_t count);
// Dense i32 array attribute (DenseI32ArrayAttr) — used e.g. for
// `llvm.getelementptr`'s `rawConstantIndices`.
MLIR_AttributeHandle MLIR_CreateAttributeDenseI32Array(MLIR_Context *ctx, string name,
                                                        const int32_t *values, size_t count);
MLIR_AttributeHandle MLIR_CreateAttributeDict(MLIR_Context *ctx, string name, MLIR_AttributeHandle *elements, size_t count);
// TypeAttr: an attribute that wraps a Type (e.g. func.func's
// `function_type` attribute, which wraps a FunctionType).
MLIR_AttributeHandle MLIR_CreateAttributeType(MLIR_Context *ctx, string name, MLIR_TypeHandle type);

// SymbolRef attribute (e.g. `callee = @foo` on func.call). The `value`
// must be the bare symbol name (no leading `@`).
MLIR_AttributeHandle MLIR_CreateAttributeSymbolRef(MLIR_Context *ctx, string name, string value);

// -----------------------------------------------------------------------------
// IR mutation primitives (used by lowering passes).
// -----------------------------------------------------------------------------

// Replace every use of `old_value` with `new_value` everywhere in the
// module that contains it. Operand lists, successor operand lists, and
// any other places the value appears are updated. Both values must have
// compatible types (the API does not check). After this call the old
// value is unused and the op that produced it can typically be erased
// with MLIR_EraseOp.
//
// Implementation note: the upstream backend uses
// mlir::Value::replaceAllUsesWith. The native backend walks the module
// linearly. In both cases the op definitions and value handles
// themselves remain valid; only operand references move.
void MLIR_ReplaceAllUsesOfValue(MLIR_Context *ctx,
                                MLIR_ValueHandle old_value,
                                MLIR_ValueHandle new_value);

// Remove `op` from its parent block (if any) and detach it. The op is
// not freed (the API leaves storage to the arena/allocator). Operands
// to and results from `op` are not touched; callers should typically
// have already called MLIR_ReplaceAllUsesOfValue on each result.
void MLIR_EraseOp(MLIR_Context *ctx, MLIR_OpHandle op);

// Replace one of `op`'s regions with `region`. The previously attached
// region (if any) is detached and abandoned. Used when a lowering
// rebuilds a function body wholesale rather than mutating it in place.
void MLIR_SetOpRegion(MLIR_Context *ctx, MLIR_OpHandle op, size_t idx,
                      MLIR_RegionHandle region);

// Detach `op`'s region at `idx` and return it as a fresh handle. After
// this call, `op` has an empty region at `idx`. The returned handle can
// be passed as part of a new MLIR_CreateOp call (or to MLIR_SetOpRegion)
// to graft the body onto a replacement op without copying.
MLIR_RegionHandle MLIR_TakeOpRegion(MLIR_Context *ctx, MLIR_OpHandle op,
                                    size_t idx);

// Returns the block that contains `op`, or MLIR_INVALID_HANDLE if the
// op is not currently attached to a block.
MLIR_BlockHandle MLIR_GetOpParentBlock(MLIR_OpHandle op);

// Returns the 0-indexed position of `op` within `block`'s operations
// list, or SIZE_MAX if `op` is not in `block`.
size_t MLIR_GetBlockOpIndex(MLIR_BlockHandle block, MLIR_OpHandle op);

// Returns the region that contains `block`, or MLIR_INVALID_HANDLE if
// the block is not currently attached to a region.
MLIR_RegionHandle MLIR_GetBlockParentRegion(MLIR_BlockHandle block);

// Detaches `op` from its current parent block (if any) and appends it
// to the end of `dest`. Operand/result handles, attributes, and regions
// of `op` are unchanged.
void MLIR_MoveOpToBlockEnd(MLIR_Context *ctx, MLIR_OpHandle op,
                           MLIR_BlockHandle dest);

// Detaches `block` from its current parent region (if any) and appends
// it to the end of `dest`. The block's ops and arguments are preserved.
void MLIR_MoveBlockToRegionEnd(MLIR_Context *ctx, MLIR_BlockHandle block,
                               MLIR_RegionHandle dest);

// Introspection
typedef enum {
    MLIR_ATTR_KIND_INTEGER,
    MLIR_ATTR_KIND_FLOAT,
    MLIR_ATTR_KIND_STRING,
    MLIR_ATTR_KIND_BOOL,
    MLIR_ATTR_KIND_ARRAY,
    MLIR_ATTR_KIND_DICT,
    MLIR_ATTR_KIND_TYPE,
    MLIR_ATTR_KIND_OTHER
} MLIR_AttrKind;

MLIR_AttrKind MLIR_GetAttributeKind(MLIR_AttributeHandle attr);
string MLIR_GetAttributeName(MLIR_AttributeHandle attr);
// Print any attribute as a string (e.g. "0 : i64", "@callee", "[1, 2]").
// Used as a fallback when the kind is MLIR_ATTR_KIND_OTHER.
string MLIR_GetAttributeAsString(MLIR_Context *ctx, MLIR_AttributeHandle attr);
int64_t MLIR_GetAttributeInteger(MLIR_AttributeHandle attr);
double MLIR_GetAttributeFloat(MLIR_AttributeHandle attr);
// For Integer and Float attributes, returns the numeric type. For other
// attribute kinds, returns MLIR_INVALID_HANDLE.
MLIR_TypeHandle MLIR_GetAttributeType(MLIR_AttributeHandle attr);
bool MLIR_GetAttributeBool(MLIR_AttributeHandle attr);
string MLIR_GetAttributeString(MLIR_AttributeHandle attr);
size_t MLIR_GetAttributeArraySize(MLIR_AttributeHandle attr);
MLIR_AttributeHandle MLIR_GetAttributeArrayElement(MLIR_AttributeHandle attr, size_t idx);
size_t MLIR_GetAttributeDictSize(MLIR_AttributeHandle attr);
MLIR_AttributeHandle MLIR_GetAttributeDictElement(MLIR_AttributeHandle attr, size_t idx);
// For TypeAttr, returns the wrapped Type. For other kinds returns
// MLIR_INVALID_HANDLE.
MLIR_TypeHandle MLIR_GetAttributeTypeValue(MLIR_AttributeHandle attr);

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

// Public MLIR C API declarations without exposing internal data structures.
// Implementations can back these APIs with different MLIR representations.
//
// This is the core IR API, which both backends (mlir_api_impl.c and
// mlir_api_impl_upstream.cpp) implement. Related headers:
//   mlir_passes.h        passes and translations built on this API
//   mlir_parser.h        the in-tree text parser
//   mlir_printer.h       the in-tree text printers
//   mlir_api_upstream.h  entry points only the upstream backend implements

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

// The context (see "Context" below). Opaque: create it with
// MLIR_CreateContext.
typedef struct MLIR_Context MLIR_Context;

// -----------------------------------------------------------------------------
// Operation kinds
// -----------------------------------------------------------------------------
//
// An op's identity is its MLIR_OpType (doc/mlir_api_design.md section 2.5).
// Names exist for printing, parsing and diagnostics only: no op stores one,
// and no function but the three below takes one.
//
// The static types are generated from mlir_ops.def. A context gives every
// other name a dynamic type, a value >= OP_TYPE_COUNT that is stable for the
// life of the context (MLIR_RegisterOpType). OP_TYPE_UNREGISTERED is only
// the lookup-miss value: no op has it.
//
// The ops of a dialect D have consecutive values, MLIR_DIALECT_<D>_FIRST to
// MLIR_DIALECT_<D>_LAST, so MLIR_OP_TYPE_IN_DIALECT(t, D) is one comparison.
// Dynamic types belong to no dialect range. (The MLIR_DIALECT_<D>_BEGIN_
// and _END_ enumerators only produce the ranges; do not use them.)

typedef enum {
    OP_TYPE_UNREGISTERED = 0,
#define MLIR_OP(enumerator, name) enumerator,
#define MLIR_DIALECT_BEGIN(D) MLIR_DIALECT_##D##_FIRST, MLIR_DIALECT_##D##_BEGIN_ = MLIR_DIALECT_##D##_FIRST - 1,
#define MLIR_DIALECT_END(D) MLIR_DIALECT_##D##_END_, MLIR_DIALECT_##D##_LAST = MLIR_DIALECT_##D##_END_ - 1,
#include "mlir_ops.def"
    OP_TYPE_COUNT
} MLIR_OpType;

// True if `t` is an op type of dialect `D`, a dialect name of mlir_ops.def
// (BUILTIN, ARITH, CF, FUNC, SCF, LLVM, WASMSSA, WASMSTACK, AARCH64).
#define MLIR_OP_TYPE_IN_DIALECT(t, D) \
    ((unsigned)(t) - (unsigned)MLIR_DIALECT_##D##_FIRST <= \
     (unsigned)(MLIR_DIALECT_##D##_LAST - MLIR_DIALECT_##D##_FIRST))
#ifndef __TINYC__
// An empty dialect block would make the range test above always true (LAST
// is FIRST - 1), so each block must hold an op: a negative array size
// otherwise.
#define MLIR_DIALECT_END(D) \
    typedef char mlir_dialect_##D##_is_not_empty[ \
        (MLIR_DIALECT_##D##_LAST >= MLIR_DIALECT_##D##_FIRST) ? 1 : -1];
#include "mlir_ops.def"
#endif

// Returns the type of `name`: its static type, the dynamic type `ctx` gave
// it before, or a new dynamic type. `name` is copied; it must not be empty.
// A few static types have a second name, an alias (`builtin.module` for
// OP_TYPE_MODULE, the upstream name of the op); an alias gives its static
// type, and MLIR_GetOpTypeName still returns the first name.
// Dynamic types are numbered per context from OP_TYPE_COUNT up, so the same
// value can be another type in another context.
MLIR_OpType MLIR_RegisterOpType(MLIR_Context *ctx, string name);
// Like MLIR_RegisterOpType, but returns OP_TYPE_UNREGISTERED instead of
// registering a name `ctx` does not know.
MLIR_OpType MLIR_LookupOpType(MLIR_Context *ctx, string name);
// The name of `type`, borrowed from `ctx`; "" for OP_TYPE_UNREGISTERED and
// values that are not types of `ctx`.
string MLIR_GetOpTypeName(MLIR_Context *ctx, MLIR_OpType type);

// -----------------------------------------------------------------------------
// Context
// -----------------------------------------------------------------------------
//
// The context owns everything that is not IR: the uniqued types (in a
// persistent arena of its own), the diagnostic handler and, on the upstream
// backend, the mlir::MLIRContext and every cache the backend keeps. Neither
// backend has process-wide mutable state, so contexts are independent of each
// other.
//
// IR (ops, regions, blocks, values, attributes and locations) is created in
// the context's current IR arena, which the caller owns and can change at any
// time with MLIR_SetArenaAllocator. Freeing an IR arena invalidates the IR
// created in it and nothing else: types stay valid until the context is
// destroyed, so a type handle can be used in IR built in any arena.
//
// Handles are borrowed pointers. An op handle (and the handles of its results
// and regions) is valid until the op is erased (MLIR_EraseOp), its arena is
// freed or the context is destroyed; the same holds on both backends.
// Strings returned by getters are borrowed from the object they describe.

typedef struct MLIR_ContextOptions {
    // The initial IR arena, owned by the caller. May be NULL; set one with
    // MLIR_SetArenaAllocator before creating IR.
    Arena *arena;
    // When true, op creation skips building the per-operand def-use chains
    // (native backend: UseNode allocations and the parallel operand_uses /
    // successor_operand_uses arrays). The def-use machinery is needed by
    // RAUW, use queries and MLIR_EraseOp's use check; the wasm -> wasmstack ->
    // wasmssa -> llvm -> aarch64 -> macho pipeline never queries uses, so
    // disabling it there removes a large per-op memory overhead. The upstream
    // backend always tracks uses and ignores this option.
    bool no_def_use_tracking;
} MLIR_ContextOptions;

// Creates a context. `options` may be NULL, which means all defaults (no IR
// arena yet, def-use tracking on). Like every allocation in the API, it
// aborts if memory runs out.
MLIR_Context *MLIR_CreateContext(const MLIR_ContextOptions *options);
// Destroys `ctx` and everything it owns (types, caches, the upstream
// MLIRContext). All IR created through `ctx` is dead afterwards: the upstream
// backend frees it (the context owns every op, block and region that has no
// parent, such as a module), while on the native backend it stays in the IR
// arenas, which belong to the caller.
void MLIR_DestroyContext(MLIR_Context *ctx);
void MLIR_SetArenaAllocator(MLIR_Context *ctx, Arena *arena);
Arena *MLIR_GetArenaAllocator(MLIR_Context *ctx);
// Turns def-use tracking (MLIR_ContextOptions.no_def_use_tracking) on or off
// for the ops created from now on. Ops created while it was off have no use
// lists, so use queries and RAUW must not be applied to their values.
void MLIR_SetContextDefUseTracking(MLIR_Context *ctx, bool enabled);
bool MLIR_HasContextDefUseTracking(MLIR_Context *ctx);

// -----------------------------------------------------------------------------
// Diagnostics
// -----------------------------------------------------------------------------
//
// Two kinds of errors (doc/mlir_api_design.md section 2.4):
//
// - Contract violations (a NULL handle where one is required, an index out of
//   range, a typed attribute getter applied to another kind, erasing an op
//   whose results still have uses, using an erased op) are bugs in the
//   caller. When the implementation is compiled with -DMLIR_API_CHECKS (debug
//   builds and CI) they abort with a message; otherwise they are not checked
//   and the result is undefined.
// - Recoverable failures (parse and verifier errors, failed passes,
//   constructs a backend does not support) return false or
//   MLIR_INVALID_HANDLE and report through the context's diagnostic handler.
//   On the upstream backend every mlir::Diagnostic of the context (from the
//   parser, the verifier and passes) reaches the same handler.

typedef enum {
    MLIR_DIAG_ERROR,
    MLIR_DIAG_WARNING,
    MLIR_DIAG_NOTE,
    MLIR_DIAG_REMARK
} MLIR_DiagSeverity;

// `message` is borrowed for the duration of the call. `loc` may be
// MLIR_INVALID_HANDLE.
typedef void (*MLIR_DiagnosticHandler)(void *user, MLIR_DiagSeverity severity,
                                       MLIR_LocationHandle loc, string message);
// Installs `fn` (called with `user`) as the handler of `ctx`. NULL restores
// the default handler, which prints `file:line:col: error: message` to stderr
// (other locations in their `loc(...)` form; `error: message` without one).
void MLIR_SetDiagnosticHandler(MLIR_Context *ctx, MLIR_DiagnosticHandler fn, void *user);
// Formats a message with printf-style `fmt` and passes it to the handler.
void MLIR_EmitDiagnostic(MLIR_Context *ctx, MLIR_DiagSeverity severity,
                         MLIR_LocationHandle loc, const char *fmt, ...);

// -----------------------------------------------------------------------------
// Operation API
// -----------------------------------------------------------------------------

// Creation & structural mutation
//
// `type` is the op's identity: a static type, or a dynamic type of `ctx`
// (MLIR_RegisterOpType) for any other op name. It must not be
// OP_TYPE_UNREGISTERED. MLIR_API_CHECKS catches a value that is not a type
// of `ctx`; a dynamic type of another context is caught only when its value
// is beyond the types `ctx` has, because the values are per-context
// counters.
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
    int64_t source_line_start);
// Like MLIR_CreateOp, but with explicit successor blocks and per-successor
// operand arrays. Used by branch ops (cf.br, cf.cond_br, ...). The successor
// blocks must be valid block handles (forward-referenced blocks are fine as
// long as MLIR_CreateBlock has been called for them).
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
    int64_t source_line_start);
void MLIR_AppendOpAttribute(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_AttributeHandle attr);

// Accessors
// The op's type, as created. For its name use
// MLIR_GetOpTypeName(ctx, MLIR_GetOpType(op)).
MLIR_OpType MLIR_GetOpType(MLIR_OpHandle op);
MLIR_LocationHandle MLIR_GetOpLocation(MLIR_OpHandle op);
size_t MLIR_GetOpNumOperands(MLIR_OpHandle op);
MLIR_ValueHandle MLIR_GetOpOperand(MLIR_OpHandle op, size_t idx);
size_t MLIR_GetOpNumResults(MLIR_OpHandle op);
MLIR_ValueHandle MLIR_GetOpResult(MLIR_OpHandle op, size_t idx);
size_t MLIR_GetOpNumResultTypes(MLIR_OpHandle op);
MLIR_TypeHandle MLIR_GetOpResultType(MLIR_OpHandle op, size_t idx);
size_t MLIR_GetOpNumAttributes(MLIR_OpHandle op);
MLIR_AttributeHandle MLIR_GetOpAttribute(MLIR_OpHandle op, size_t idx);
// Like MLIR_GetOpAttribute, but looks up by name including inherent
// attributes that are stored in the op's Properties (e.g. LLVM::CallOp
// `callee` and `var_callee_type`). Returns MLIR_INVALID_HANDLE on miss.
MLIR_AttributeHandle MLIR_GetOpAttributeByName(MLIR_OpHandle op, const char *name);
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
// Insert `op` at position `idx` in `block`'s op list, shifting later ops
// down by one. `idx` is clamped to [0, n_ops].
void MLIR_InsertBlockOpAtIndex(MLIR_Context *ctx, MLIR_BlockHandle block, MLIR_OpHandle op, size_t idx);
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
    MLIR_VALUE_BLOCK_ARG,
    MLIR_VALUE_OP_RESULT
} MLIR_ValueKind;

// Creation
MLIR_ValueHandle MLIR_CreateValueBlockArg(MLIR_Context *ctx, string register_name, uint32_t result_index, MLIR_TypeHandle type, MLIR_LocationHandle location);
MLIR_ValueHandle MLIR_CreateValueOpResult(MLIR_Context *ctx, MLIR_OpHandle def, uint32_t result_index, MLIR_TypeHandle type, string register_name, MLIR_LocationHandle location);

// Accessors
MLIR_ValueKind MLIR_GetValueKind(MLIR_ValueHandle value);
MLIR_TypeHandle MLIR_GetValueType(MLIR_ValueHandle value);
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
MLIR_TypeHandle MLIR_CreateTypeTensor(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type);
MLIR_TypeHandle MLIR_CreateTypeMemref(MLIR_Context *ctx, const int64_t *shape, size_t rank, MLIR_TypeHandle element_type);

// LLVM dialect types — used by frontends that emit `!llvm.ptr`,
// `!llvm.struct`, and `!llvm.array`. Identified structs are mutable: create
// with MLIR_CreateTypeLLVMStructIdentified, then call
// MLIR_SetTypeLLVMStructBody once. Recursion through `!llvm.ptr` (which is
// opaque) is fully supported because the body never mentions the recursive
// struct's body type.
MLIR_TypeHandle MLIR_CreateTypeLLVMPointer(MLIR_Context *ctx);
MLIR_TypeHandle MLIR_CreateTypeLLVMPointerInAddressSpace(MLIR_Context *ctx, uint32_t address_space);
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
// Aggregate-array global with a raw-bytes initializer:
//   llvm.mlir.global internal @<sym>("<bytes>") : !llvm.array<N x iX>
// `bytes` is the packed little-endian initializer (size must equal
// N * sizeof(elem)). Backends interpret it as raw bytes.
MLIR_OpHandle MLIR_CreateLLVMGlobalArrayInit(MLIR_Context *ctx,
                                              string sym_name,
                                              MLIR_TypeHandle array_ty,
                                              bool is_constant,
                                              string bytes,
                                              MLIR_LocationHandle loc);

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

// Introspection & formatting
bool MLIR_IsTypeInteger(MLIR_TypeHandle type);
bool MLIR_IsTypeFloat(MLIR_TypeHandle type);
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
// LLVM struct/array type introspection. Used by lowering passes that
// need to compute byte offsets and sizes (e.g. native LLVM->WASM).
bool            MLIR_IsTypeLLVMStruct(MLIR_TypeHandle type);
size_t          MLIR_GetTypeLLVMStructNumFields(MLIR_TypeHandle type);
MLIR_TypeHandle MLIR_GetTypeLLVMStructField(MLIR_TypeHandle type, size_t idx);
bool            MLIR_IsTypeLLVMArray(MLIR_TypeHandle type);
MLIR_TypeHandle MLIR_GetTypeLLVMArrayElement(MLIR_TypeHandle type);
uint64_t        MLIR_GetTypeLLVMArrayNumElements(MLIR_TypeHandle type);
uint32_t        MLIR_GetTypeLLVMPointerAddressSpace(MLIR_TypeHandle type);
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
// Dense i64 array attribute (DenseI64ArrayAttr) — used e.g. for
// `scf.index_switch`'s `cases` attribute.
MLIR_AttributeHandle MLIR_CreateAttributeDenseI64Array(MLIR_Context *ctx, string name,
                                                        const int64_t *values, size_t count);
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

// Erases `op`: removes it from its parent block (if any), drops its operand
// uses and destroys it together with the ops nested in its regions. Nothing
// outside the erased subtree may still use a value or a block defined inside
// it: not its results, and not the results, block arguments or blocks nested
// in its regions. That includes uses held by detached or never-inserted ops
// and by blocks removed with MLIR_EraseBlock. Replace or erase those uses
// first (e.g. with MLIR_ReplaceAllUsesOfValue). On upstream such a use would
// point into freed memory; MLIR_API_CHECKS checks it (natively for values
// with def-use tracking). Afterwards the handles of `op`, its results
// and its regions are dead on both backends; using them is a contract
// violation (checked with MLIR_API_CHECKS), including asking
// MLIR_GetOpParentBlock whether the op was erased. The native backend leaves
// the storage to the arena.
void MLIR_EraseOp(MLIR_Context *ctx, MLIR_OpHandle op);

// Removes `op` from its parent block (if any) and keeps it: the handle stays
// valid, and its operands, results, attributes and regions are unchanged.
// Insert it again with MLIR_AppendBlockOp / MLIR_InsertBlockOpAtIndex, or
// erase it with MLIR_EraseOp. Does nothing for an op without a parent block.
void MLIR_DetachOp(MLIR_Context *ctx, MLIR_OpHandle op);

// Detach `op`'s region at `idx` and return it as a fresh handle. After
// this call, `op` has an empty region at `idx`. The returned handle can
// be passed as part of a new MLIR_CreateOp call to graft the body onto a
// replacement op without copying.
MLIR_RegionHandle MLIR_TakeOpRegion(MLIR_Context *ctx, MLIR_OpHandle op,
                                    size_t idx);

// Returns the block that contains `op`, or MLIR_INVALID_HANDLE if the
// op is not currently attached to a block.
MLIR_BlockHandle MLIR_GetOpParentBlock(MLIR_OpHandle op);

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

// Replace operand `idx` of `op` with `value`. The previous operand
// reference is dropped (use lists are updated transparently by the
// upstream backend; the native backend simply overwrites the slot).
void MLIR_SetOpOperand(MLIR_Context *ctx, MLIR_OpHandle op,
                       size_t idx, MLIR_ValueHandle value);

// Replace successor `succ_idx` of `op` with `block`. `op` must be a
// terminator with at least `succ_idx + 1` successors. This does not
// touch the successor operands; use MLIR_SetOpSuccessorOperands to
// rewrite them.
void MLIR_SetOpSuccessor(MLIR_Context *ctx, MLIR_OpHandle op,
                         size_t succ_idx, MLIR_BlockHandle block);

// Replace the entire successor-operand list at `succ_idx` of `op` with
// the given `values[0..n)`. On the upstream backend, successor operands
// share storage with regular operands via BranchOpInterface; this
// function resizes that segment so the count after the call is exactly
// `n`. On the native backend, the per-successor operand vector is
// reallocated.
void MLIR_SetOpSuccessorOperands(MLIR_Context *ctx, MLIR_OpHandle op,
                                 size_t succ_idx,
                                 const MLIR_ValueHandle *values, size_t n);

// Detach `block` from its parent region (if any). The block and its ops are
// not destroyed: they stay until the context is destroyed. No branch outside
// the block may still target it (checked with MLIR_API_CHECKS). Its values
// may still be used elsewhere, e.g. by other unreachable blocks that are
// erased in a later round; but before an op whose values the block's ops use
// is erased, those uses must be gone (MLIR_EraseOp).
void MLIR_EraseBlock(MLIR_Context *ctx, MLIR_BlockHandle block);

// Insert `block` into `region` immediately after `after`. If `after`
// is MLIR_INVALID_HANDLE the block is inserted at the start of the
// region. `block` must not already belong to a region; otherwise it is
// detached from its current parent first.
void MLIR_InsertRegionBlockAfter(MLIR_Context *ctx, MLIR_RegionHandle region,
                                 MLIR_BlockHandle block, MLIR_BlockHandle after);

// Insert `block` into `region` immediately before `before`. If
// `before` is MLIR_INVALID_HANDLE the block is appended to the region.
// `block` must not already belong to a region; otherwise it is
// detached from its current parent first.
void MLIR_InsertRegionBlockBefore(MLIR_Context *ctx, MLIR_RegionHandle region,
                                  MLIR_BlockHandle block, MLIR_BlockHandle before);

// Returns the last operation in `block` (its terminator), or
// MLIR_INVALID_HANDLE if the block is empty.
MLIR_OpHandle MLIR_GetBlockTerminator(MLIR_BlockHandle block);

// Returns the op that contains the parent region of `block`. If the
// region is a top-level (orphan) region, returns MLIR_INVALID_HANDLE.
MLIR_OpHandle MLIR_GetBlockParentOp(MLIR_BlockHandle block);

// Returns true if `block` is the first block of its parent region.
bool MLIR_IsBlockEntry(MLIR_BlockHandle block);

// Returns the number of incoming edges to `block` from terminators in
// other blocks of the same parent region. Each (predecessor block,
// successor slot) pair is counted once even if a predecessor branches
// to `block` from multiple slots.
size_t MLIR_GetBlockNumPredecessors(MLIR_BlockHandle block);

// Returns the predecessor at index `idx` of `block`. The 0-indexed
// successor slot in the predecessor's terminator that points at
// `block` is written to *out_succ_idx (if non-null). Returns
// MLIR_INVALID_HANDLE if `idx` is out of range.
MLIR_BlockHandle MLIR_GetBlockPredecessor(MLIR_BlockHandle block, size_t idx,
                                          size_t *out_succ_idx);

// Adds a new block argument of type `type` to the end of `block`'s
// argument list and returns its Value handle.
MLIR_ValueHandle MLIR_AddBlockArgument(MLIR_Context *ctx, MLIR_BlockHandle block,
                                       MLIR_TypeHandle type,
                                       MLIR_LocationHandle loc);

// Erases the `count` consecutive block arguments starting at `start`.
// Each argument must have no remaining uses; the caller is responsible
// for replacing them first via MLIR_ReplaceAllUsesOfValue.
void MLIR_EraseBlockArguments(MLIR_Context *ctx, MLIR_BlockHandle block,
                              size_t start, size_t count);

// Returns the block containing the definition of `value`. For block
// arguments this is the block carrying the argument; for op results it
// is the parent block of the defining op. Returns MLIR_INVALID_HANDLE
// if the value is detached.
MLIR_BlockHandle MLIR_GetValueParentBlock(MLIR_ValueHandle value);

// Returns the number of uses of `value`. A use is a single operand
// reference (operand, successor-operand of a terminator, etc.).
size_t MLIR_GetValueNumUses(MLIR_Context *ctx, MLIR_ValueHandle value);

// Returns one use of `value`. The owning operation is returned; the
// operand index within that op is written to *out_operand_idx (if
// non-null). Uses are returned in implementation-defined order. The
// order may change if any operand of any op is mutated; callers must
// capture uses into a vector before mutating.
MLIR_OpHandle MLIR_GetValueUseOwner(MLIR_Context *ctx, MLIR_ValueHandle value,
                                    size_t idx, size_t *out_operand_idx);

// Appends `value` to the successor-operand list at `succ_idx` of `op`.
// On the upstream backend this resizes the segment via
// `BranchOpInterface::SuccessorOperands::append`; on the native backend
// the per-successor operand vector is reallocated.
void MLIR_AppendOpSuccessorOperand(MLIR_Context *ctx, MLIR_OpHandle op,
                                   size_t succ_idx, MLIR_ValueHandle value);

// Moves all ops of `src` to the end of `dst`. `src` becomes empty but
// remains attached to its parent region.
void MLIR_SpliceBlockOps(MLIR_Context *ctx, MLIR_BlockHandle dst,
                         MLIR_BlockHandle src);

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

// Accessors
MLIR_LocationKind MLIR_GetLocationKind(MLIR_LocationHandle loc);
string MLIR_GetLocationFileFilename(MLIR_LocationHandle loc);
int MLIR_GetLocationFileLine(MLIR_LocationHandle loc);
int MLIR_GetLocationFileColumn(MLIR_LocationHandle loc);
string MLIR_GetLocationName(MLIR_LocationHandle loc);

#ifdef __cplusplus
}
#endif

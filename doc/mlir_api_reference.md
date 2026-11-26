# MLIR C API Reference

This document is the reference for the public MLIR C API implemented identically by two backends:
- **Native backend** (`mlir_api_impl.c`): zero-libc, arena-allocated C implementation with 32-bit indexed handles (~150 B/op, ~35–45 ns/op creation).
- **Upstream backend** (`mlir_api_impl_upstream.cpp`): zero-copy wrapper over upstream MLIR 19 (`mlir::Operation *`, `mlir::Value::getAsOpaquePointer()`, `mlir::PassManager`).

Both backends are tested for byte-identical behavior in `tests/cross/driver.c` and `tests/acceptance/test_acceptance.c`.

---

## 1. Preface: API Contracts and Rules

1. **Context ownership and isolation**: `MLIR_Context` owns all non-IR state (uniqued types, uniqued attributes, locations, dynamic op types, dynamic attribute keys, diagnostic handler, and on upstream the `mlir::MLIRContext`). Neither backend has process-wide mutable state; contexts are independent.
2. **IR arena lifetime**: IR objects (ops, regions, blocks, values) are allocated using the context's current IR arena (`MLIR_SetArenaAllocator`). Types, attributes, and locations live for the lifetime of the `MLIR_Context`, so type and attribute handles outlive IR arenas.
3. **Opaque handles**: Handles (`MLIR_OpHandle`, `MLIR_RegionHandle`, `MLIR_BlockHandle`, `MLIR_ValueHandle`, `MLIR_TypeHandle`, `MLIR_AttributeHandle`, `MLIR_LocationHandle`, `MLIR_UseHandle`, `MLIR_PredHandle`) are opaque `uintptr_t` values (`0 == MLIR_INVALID_HANDLE`). Compare handles only for equality; never order or dereference them.
4. **Uniqued types and attributes**: Equal types (within the same context) share one `MLIR_TypeHandle`; equal attributes (except `MLIR_CreateAttributeDistinct`) share one `MLIR_AttributeHandle`. Equality is handle comparison (`t1 == t2`, `a1 == a2`).
5. **Erase semantics**: `MLIR_EraseOp` destroys an op and its nested regions/ops. Nothing outside the erased subtree may still use any value or block defined inside it. Replace or erase external uses first (e.g. via `MLIR_ReplaceAllUsesOfValue`).
6. **Errors vs contract violations**:
   - **Contract violations** (NULL handle where required, out-of-range index, wrong attribute kind, erasing an op with live external uses, cross-context handle) abort with a message when built with `-DMLIR_API_CHECKS`.
   - **Recoverable failures** (verification errors, failed passes, parse errors) return `false` or `MLIR_INVALID_HANDLE` and report through the context's diagnostic handler (`MLIR_SetDiagnosticHandler`).
7. **Borrowed strings**: `string` values returned by getters (`MLIR_GetOpTypeName`, `MLIR_GetAttrKeyName`, `MLIR_GetAttributeString`, `MLIR_GetCallee`, `MLIR_GetFunctionPassthrough`) are borrowed from the context or object.

---

## 2. Header Map

| Header | Contents |
|---|---|
| `mlir_api.h` | Core IR API (contexts, op types, traits, verification, builder, ops/regions/blocks/values, O(1) iterators, walks, CFG/dominators, types, attributes, LLVM-level completeness, locations) |
| `mlir_passes.h` | Pass and pipeline infrastructure (`MLIR_Pass`, `MLIR_RunPasses`) and in-tree lowering/translation/optimization passes |
| `mlir_parser.h` | In-tree MLIR text parser and parser-only helpers |
| `mlir_printer.h` | In-tree classic and generic MLIR text printers |
| `mlir_api_upstream.h` | Upstream-only entry points (e.g. `MLIR_LowerToLLVMDialectUpstream`, `MLIR_TranslateModuleToLLVMIRUpstream`, `MLIR_TranslateModuleToWasmUpstream`) |

---

## 3. Core API Reference (`mlir_api.h`)

### 3.1 Context and Diagnostics
- `MLIR_Context *MLIR_CreateContext(const MLIR_ContextOptions *options)` / `void MLIR_DestroyContext(MLIR_Context *ctx)`
  - `MLIR_ContextOptions`: `{ Arena *arena; bool no_def_use_tracking; size_t expected_ops; }`.
- `void MLIR_SetArenaAllocator(MLIR_Context *ctx, Arena *arena)` / `Arena *MLIR_GetArenaAllocator(MLIR_Context *ctx)`
- `void MLIR_SetContextDefUseTracking(MLIR_Context *ctx, bool enabled)` / `bool MLIR_HasContextDefUseTracking(MLIR_Context *ctx)`
- `void MLIR_SetContextExpectedOps(MLIR_Context *ctx, size_t n)`
- `void MLIR_SetDiagnosticHandler(MLIR_Context *ctx, MLIR_DiagnosticHandler fn, void *user)`
- `void MLIR_EmitDiagnostic(MLIR_Context *ctx, MLIR_DiagSeverity severity, MLIR_LocationHandle loc, const char *fmt, ...)`

### 3.2 Op Identity, Traits, and Verification
Static op types are generated from `mlir_ops.def` (`OP_TYPE_*`), grouped into contiguous dialect ranges tested by `MLIR_OP_TYPE_IN_DIALECT(t, D)` (`BUILTIN`, `ARITH`, `CF`, `FUNC`, `SCF`, `LLVM`, `WASMSSA`, `WASMSTACK`, `AARCH64`).
- `MLIR_OpType MLIR_RegisterOpType(MLIR_Context *ctx, string name)`
- `MLIR_OpType MLIR_LookupOpType(MLIR_Context *ctx, string name)`
- `string MLIR_GetOpTypeName(MLIR_Context *ctx, MLIR_OpType type)`
- `MLIR_OpTraits MLIR_GetOpTypeTraits(MLIR_OpType type)` / `bool MLIR_OpHasTrait(MLIR_Context *ctx, MLIR_OpHandle op, MLIR_OpTraits traits)`
  - Trait bits: `MLIR_T_TERMINATOR`, `MLIR_T_BRANCH`, `MLIR_T_RETURN_LIKE`, `MLIR_T_PURE`, `MLIR_T_MEM_READ`, `MLIR_T_MEM_WRITE`, `MLIR_T_COMMUTATIVE`, `MLIR_T_CONSTANT`, `MLIR_T_ELEMENTWISE`, `MLIR_T_FASTMATH`, `MLIR_T_CALL`, `MLIR_T_FUNCTION`, `MLIR_T_SYMBOL`, `MLIR_T_ISOLATED`, `MLIR_T_NO_TERMINATOR`, `MLIR_T_SINGLE_BLOCK`, `MLIR_T_ZERO_RESULT`, `MLIR_T_ONE_RESULT`.
- `bool MLIR_VerifyOp(MLIR_Context *ctx, MLIR_OpHandle op)` — verifies `op` and nested regions (on upstream delegates to `mlir::verify`).

### 3.3 Attribute Keys and Attributes
Static keys are generated from `mlir_attr_keys.def` (`MLIR_KEY_*`).
- `MLIR_AttrKey MLIR_InternAttrKey(MLIR_Context *ctx, string name)` / `MLIR_LookupAttrKey` / `MLIR_GetAttrKeyName`
- `MLIR_NamedAttr MLIR_Named(MLIR_AttrKey key, MLIR_AttributeHandle value)`
- Constructors: `MLIR_CreateAttributeInteger`, `MLIR_CreateAttributeFloat`, `MLIR_CreateAttributeBool`, `MLIR_CreateAttributeString`, `MLIR_CreateAttributeSymbolRef`, `MLIR_CreateAttributeType`, `MLIR_CreateAttributeArray`, `MLIR_CreateAttributeDict`, `MLIR_CreateAttributeDenseI32Array`, `MLIR_CreateAttributeDenseI64Array`, `MLIR_CreateAttributeDistinct`, `MLIR_CreateAttributeLLVMLinkageInternal`, `MLIR_CreateAttributeFastMath`, `MLIR_CreateAttributeSplat`.
- Introspection: `MLIR_GetAttributeKind` (`MLIR_ATTR_KIND_*`), `MLIR_GetAttributeInteger`, `MLIR_GetAttributeFloat`, `MLIR_GetAttributeBool`, `MLIR_GetAttributeString`, `MLIR_GetAttributeAsString`, `MLIR_GetAttributeType`, `MLIR_GetAttributeTypeValue`, `MLIR_GetAttributeArraySize` / `Element`, `MLIR_GetAttributeDictSize` / `Key` / `Element`, `MLIR_GetAttributeDenseSize` / `Element`, `MLIR_GetAttributeFastMath`.

### 3.4 Types
- Constructors: `MLIR_CreateTypeInteger`, `MLIR_CreateTypeFloat`, `MLIR_CreateTypeIndex`, `MLIR_CreateTypeVector`, `MLIR_CreateTypeTensor`, `MLIR_CreateTypeMemref`, `MLIR_CreateTypeFunction`, `MLIR_CreateTypeLLVMPointer`, `MLIR_CreateTypeLLVMPointerInAddressSpace`, `MLIR_CreateTypeLLVMVoid`, `MLIR_CreateTypeLLVMArray`, `MLIR_CreateTypeLLVMStructIdentified` + `MLIR_SetTypeLLVMStructBody`, `MLIR_CreateTypeLLVMFunction`.
- Introspection: `MLIR_GetTypeKind` (`MLIR_TYPE_KIND_*`), `MLIR_GetTypeWidth`, `MLIR_GetTypeElement`, `MLIR_GetTypeNumElements`, `MLIR_GetTypeShapeRank` / `Dim`, `MLIR_GetTypeFunctionNumInputs` / `Input` / `NumResults` / `Result` / `IsVarArg`, `MLIR_GetTypeLLVMStructNumFields` / `Field` / `Name`, `MLIR_IsTypeLLVMStructOpaque`, `MLIR_GetTypeLLVMPointerAddressSpace`, `MLIR_GetTypeString`.

### 3.5 Op Construction and Builder
- `MLIR_OpHandle MLIR_CreateOpFromSpec(MLIR_Context *ctx, const MLIR_OpSpec *spec)` — creates a detached op; results are created from `spec->result_types`; successor operands are laid out flat at the tail of `spec->operands` according to `spec->successor_operand_counts`.
- `void MLIR_InsertOpBefore(MLIR_Context *ctx, MLIR_BlockHandle block, MLIR_OpHandle op, MLIR_OpHandle before)`
- `MLIR_Builder`: `MLIR_BuilderAtEnd(ctx, block)`, `MLIR_BuilderBefore(ctx, op)`, `MLIR_BuilderAfter(ctx, op)`, `MLIR_BuildOp(&b, &spec)`.
- Shorthands: `MLIR_BuildBinary`, `MLIR_BuildUnary`, `MLIR_BuildFBinary`, `MLIR_BuildFUnary`, `MLIR_BuildConstantInt` (scalar or vector splat), `MLIR_BuildConstantFloat` (scalar or vector splat), `MLIR_BuildUndef`, `MLIR_BuildPoison`, `MLIR_BuildZero`, `MLIR_BuildCallIntrinsic`, `MLIR_BuildBr`, `MLIR_BuildCondBr`.

### 3.6 O(1) Traversal, Walks, and CFG Info
Prefer O(1) iterators over indexed loops (`GetBlockOp(i)`, `GetRegionBlock(i)`, `GetValueUseOwner(i)`, `GetBlockPredecessor(i)`):
- Ops in a block: `MLIR_GetBlockFirstOp`, `MLIR_GetBlockLastOp`, `MLIR_GetOpNext`, `MLIR_GetOpPrev`, `MLIR_GetParentOp`. For read-only hot loops over a block's ops, prefer the cursor `MLIR_BlockOpIter it = MLIR_BlockOpsBegin(ctx, blk); while ((op = MLIR_BlockOpIterNext(ctx, &it)) != MLIR_INVALID_HANDLE)` which avoids per-step parent-block lookups on both backends.
- Blocks in a region: `MLIR_GetRegionFirstBlock`, `MLIR_GetRegionLastBlock`, `MLIR_GetBlockNext`, `MLIR_GetBlockPrev`.
- Uses of a value: `MLIR_GetValueFirstUse`, `MLIR_GetUseNext`, `MLIR_GetUseOwner`, `MLIR_GetUseOperandIndex`, `MLIR_ValueHasOneUse`, `MLIR_IsValueUnused`.
- Predecessors of a block: `MLIR_GetBlockFirstPred`, `MLIR_GetPredNext`, `MLIR_GetPredBlock`, `MLIR_GetPredTerminator`, `MLIR_GetPredSuccessorIndex`, `MLIR_GetIncomingValue`, `MLIR_SetIncomingValue`.
- Recursive walk: `MLIR_WalkResult MLIR_Walk(MLIR_Context *ctx, MLIR_OpHandle root, MLIR_WalkOrder order, MLIR_WalkFn fn, void *user)` (`MLIR_WALK_PRE` / `MLIR_WALK_POST`; safe to erase the visited op in `MLIR_WALK_POST`).
- Block ordering & dominators: `MLIR_GetBlockOrder(ctx, region, MLIR_ORDER_RPO | MLIR_ORDER_POSTORDER, scratch, &out)`, `MLIR_ComputeCFGInfo(ctx, region, scratch)` (returns RPO, Cooper–Harvey–Kennedy immediate dominators, natural loop depths & loop headers), `MLIR_CFGGetRPOIndex`, `MLIR_CFGDominates`.

### 3.7 IR Mutation
- `MLIR_SetOpAttr`, `MLIR_SetOpOperand`, `MLIR_SetOpSuccessor`, `MLIR_SetOpSuccessorOperands`, `MLIR_AppendOpSuccessorOperand`
- `MLIR_ReplaceAllUsesOfValue`, `MLIR_EraseOp`, `MLIR_DetachOp`, `MLIR_TakeOpRegion`, `MLIR_MoveOpToBlockEnd`, `MLIR_SpliceBlockOps`
- `MLIR_AddBlockArgument`, `MLIR_EraseBlockArguments`, `MLIR_EraseBlock`, `MLIR_InsertRegionBlockBefore`, `MLIR_InsertRegionBlockAfter`, `MLIR_MoveBlockToRegionEnd`

### 3.8 LLVM-Level Completeness
- Fast-math flags (`MLIR_FastMath`): `MLIR_FMF_NONE`, `MLIR_FMF_NNAN`, `MLIR_FMF_NINF`, `MLIR_FMF_NSZ`, `MLIR_FMF_ARCP`, `MLIR_FMF_CONTRACT`, `MLIR_FMF_AFN`, `MLIR_FMF_REASSOC`, `MLIR_FMF_FAST`; `MLIR_GetOpFastMath`, `MLIR_SetOpFastMath`.
- Constant value matchers: `MLIR_MatchConstantFloat`, `MLIR_MatchConstantInt`, `MLIR_IsConstantZero` (`lane < 0` matches scalar or splat; `lane >= 0` queries a specific lane).
- Calls & functions: `MLIR_GetCallee`, `MLIR_IsFunctionDeclaration`, `MLIR_GetFunctionPassthrough`, `MLIR_HasFunctionPassthrough`, `MLIR_SetFunctionPassthrough`.
- Target info: `MLIR_TargetInfo`, `MLIR_TargetFromTriple`, `MLIR_TargetIsX86`, `MLIR_TargetHasFeature` (honors per-function `target_features` / `passthrough` `"target-features"` overrides).

### 3.9 Pass and Pipeline Infrastructure (`mlir_passes.h`)
- `MLIR_PassContext { MLIR_Context *ctx; const MLIR_TargetInfo *target; Arena *scratch; }`
- `MLIR_Pass { string name; MLIR_PassAnchor anchor; MLIR_PassFn run; void *state; }` (`MLIR_ANCHOR_MODULE` or `MLIR_ANCHOR_FUNCTION`).
- `MLIR_PipelineOptions { const MLIR_TargetInfo *target; bool verify_each; bool timing; string print_after; }`
- `bool MLIR_RunPasses(MLIR_Context *ctx, MLIR_OpHandle module, const MLIR_Pass *passes, size_t n, const MLIR_PipelineOptions *opts)`
- In-tree pass constructors: `MLIR_PassMem2Reg()`, `MLIR_PassLoadCSE()`, `MLIR_PassArithGVN()`, `MLIR_PassDCE()`, `MLIR_PassLowerToLLVMDialect()`, `MLIR_PassLiftCfToScf(options)`.

---

## 4. Examples

### 4.1 Writing a Function-Anchored Optimization Pass

```c
#include "mlir_api.h"
#include "mlir_passes.h"

// Fold `x + 0.0` (with fast-math) -> `x` in every function body.
static bool fold_add_zero_run(MLIR_PassContext *pc, MLIR_OpHandle fn, void *state) {
    (void)state;
    MLIR_Context *ctx = pc->ctx;
    MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, fn, 0);
    for (MLIR_BlockHandle b = MLIR_GetRegionFirstBlock(ctx, r); b; b = MLIR_GetBlockNext(ctx, b)) {
        for (MLIR_OpHandle op = MLIR_GetBlockFirstOp(ctx, b); op; ) {
            MLIR_OpHandle next = MLIR_GetOpNext(ctx, op);
            if (MLIR_GetOpType(ctx, op) == OP_TYPE_LLVM_FADD &&
                (MLIR_GetOpFastMath(ctx, op) & MLIR_FMF_NSZ)) {
                MLIR_ValueHandle lhs = MLIR_GetOpOperand(ctx, op, 0);
                MLIR_ValueHandle rhs = MLIR_GetOpOperand(ctx, op, 1);
                if (MLIR_IsConstantZero(ctx, rhs, -1)) {
                    MLIR_ReplaceAllUsesOfValue(ctx, MLIR_GetOpResult(ctx, op, 0), lhs);
                    MLIR_EraseOp(ctx, op);
                }
            }
            op = next;
        }
    }
    return true;
}

void run_example_pipeline(MLIR_Context *ctx, MLIR_OpHandle mod) {
    MLIR_Pass passes[2];
    passes[0].name = str_lit("fold-add-zero");
    passes[0].anchor = MLIR_ANCHOR_FUNCTION;
    passes[0].run = fold_add_zero_run;
    passes[0].state = NULL;
    passes[1] = MLIR_PassDCE();

    MLIR_PipelineOptions opts = {0};
    opts.verify_each = true;
    MLIR_RunPasses(ctx, mod, passes, 2, &opts);
}
```

### 4.2 Writing a Dialect Lowering

```c
// Lower `arith.addi` -> `llvm.add` in post-order walk (safe to erase visited op).
static MLIR_WalkResult lower_addi(MLIR_Context *ctx, MLIR_OpHandle op, void *user) {
    (void)user;
    if (MLIR_GetOpType(ctx, op) != OP_TYPE_ARITH_ADDI) return MLIR_WALK_ADVANCE;
    MLIR_Builder b = MLIR_BuilderBefore(ctx, op);
    MLIR_ValueHandle res = MLIR_GetOpResult(ctx, op, 0);
    MLIR_ValueHandle lowered = MLIR_BuildBinary(
        &b, OP_TYPE_LLVM_ADD, MLIR_GetValueType(ctx, res),
        MLIR_GetOpOperand(ctx, op, 0), MLIR_GetOpOperand(ctx, op, 1));
    MLIR_ReplaceAllUsesOfValue(ctx, res, lowered);
    MLIR_EraseOp(ctx, op);
    return MLIR_WALK_ADVANCE;
}

void lower_module(MLIR_Context *ctx, MLIR_OpHandle mod) {
    MLIR_Walk(ctx, mod, MLIR_WALK_POST, lower_addi, NULL);
}
```

### 4.3 Writing a Backend Codegen Loop

```c
// Emit instructions for every reachable block in RPO using O(1) iterators
// and integer-keyed attribute/type queries (zero string comparisons).
void emit_function_backend(MLIR_Context *ctx, MLIR_OpHandle fn, Arena *scratch) {
    if (MLIR_IsFunctionDeclaration(ctx, fn)) return;
    MLIR_RegionHandle r = MLIR_GetOpRegion(ctx, fn, 0);
    MLIR_CFGInfo *cfg = MLIR_ComputeCFGInfo(ctx, r, scratch);
    for (size_t i = 0; i < cfg->n_blocks; i++) {
        MLIR_BlockHandle blk = cfg->rpo[i];
        for (MLIR_OpHandle op = MLIR_GetBlockFirstOp(ctx, blk); op; op = MLIR_GetOpNext(ctx, op)) {
            switch (MLIR_GetOpType(ctx, op)) {
                case OP_TYPE_LLVM_ADD: {
                    MLIR_TypeHandle ty = MLIR_GetValueType(ctx, MLIR_GetOpResult(ctx, op, 0));
                    uint32_t width = MLIR_GetTypeWidth(ctx, ty);
                    (void)width; // emit 32-bit or 64-bit add
                    break;
                }
                case OP_TYPE_LLVM_MLIR_CONSTANT: {
                    int64_t imm = 0;
                    if (MLIR_MatchConstantInt(ctx, MLIR_GetOpResult(ctx, op, 0), -1, &imm)) {
                        (void)imm; // emit mov immediate
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }
}
```

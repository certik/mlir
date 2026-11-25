// Passes and translations built on the core API (mlir_api.h).
//
// MLIR_LowerToLLVMDialect and MLIR_TranslateModuleTo{LLVMIR,Wasm} are
// implemented in dedicated translation units (mlir_lower_to_llvm.c,
// mlir_translate_to_llvm_ir.c, mlir_translate_to_wasm.c) that use only
// mlir_api.h, with no upstream MLIR types. The same .o files link into
// both the native and the upstream builds, so these functions are always
// available and produce identical output whichever core API
// implementation is linked in. MLIR_LiftCfToScf is implemented by each
// backend on top of the shared C port in mlir_lift_cf_to_scf.c.
//
// Upstream-only siblings that run upstream MLIR's own pipeline are in
// mlir_api_upstream.h.
//
// All entry points print diagnostics to stderr on failure.

#pragma once

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lower a `builtin.module` op to the LLVM dialect by running the
// in-tree conversion (scf->cf, then arith/memref/cf/func/vector ->
// llvm). Mutates `module` in place. Returns true on success.
//
// This is the one lowering every in-tree backend takes its input from:
// the result is LLVM dialect only, with control flow as a flat
// llvm.br / llvm.cond_br CFG. Backends that need structured control
// flow (wasm) rebuild it themselves from that CFG.
bool MLIR_LowerToLLVMDialect(MLIR_Context *ctx, MLIR_OpHandle module);

// Translate a `builtin.module` op already lowered to the LLVM dialect
// into LLVM IR text (`.ll`). Returns the IR text on success or an
// empty string on failure.
string MLIR_TranslateModuleToLLVMIR(MLIR_Context *ctx, MLIR_OpHandle module);

// Translate a `builtin.module` op already lowered to the LLVM dialect
// (the output of MLIR_LowerToLLVMDialect; `module` is consumed: the
// wasm backend structurizes its CFG in place)
// into a WebAssembly relocatable object file (wasm32-wasi). Returns
// the raw object bytes on success or an empty string on failure. The
// returned bytes are NOT a runnable wasm module — they still need to
// be linked (typically with `wasm-ld`) against any runtime/imports the
// program uses, producing a final `.wasm` module that can be run by
// wasmtime. This implementation chains the in-tree native pipeline
// (mlir_llvm_to_wasmssa -> mlir_wasmssa_to_wasmstack ->
// mlir_wasmstack_to_bin).
string MLIR_TranslateModuleToWasm(MLIR_Context *ctx, MLIR_OpHandle module);

// Options for MLIR_LiftCfToScf. Zero-initialise for the defaults.
typedef struct MLIR_LiftCfToScfOptions {
    // Which dialect's branch ops form the CFG to structurize, and which
    // dialect the lift's own intermediate branches are created in:
    //   false: cf.br / cf.cond_br / cf.switch
    //   true:  llvm.br / llvm.cond_br / llvm.switch (a module that is
    //          already LLVM dialect only, e.g. MLIR_LowerToLLVMDialect
    //          output). Branch ops of the other family are rejected.
    bool llvm_cfg;
} MLIR_LiftCfToScfOptions;

// Lift the control-flow graph of every `func.func` / `llvm.func` body in
// `module` into structured control flow (scf.if / scf.while /
// scf.index_switch). `options` may be NULL (all defaults).
//
// Its one in-tree user is the wasm backend (mlir_llvm_to_wasmssa.c), which
// runs it with `llvm_cfg = true` as its internal structurization step:
// WebAssembly has no arbitrary branches, so the backend rebuilds
// structured control flow from the LLVM-dialect CFG it is handed.
//
// Returns true on success, false on failure (e.g. a CFG that the lift
// could not transform into SCF). Both API implementations run the
// in-tree agnostic C port of upstream MLIR's `transformCFGToSCF`
// (Bahmann/Reissmann 2015) in `mlir_lift_cf_to_scf.c` (see that file's
// header for the documented limitations); the upstream-backed
// implementation additionally drops unreachable blocks first.
bool MLIR_LiftCfToScf(MLIR_Context *ctx, MLIR_OpHandle module,
                      const MLIR_LiftCfToScfOptions *options);

// -----------------------------------------------------------------------------
// Pass and pipeline infrastructure (doc/mlir_api_design.md section 3.3)
// -----------------------------------------------------------------------------

typedef struct MLIR_PassContext {
    MLIR_Context          *ctx;
    const MLIR_TargetInfo *target;   // optional target info; may be NULL
    Arena                 *scratch;  // reset after each pass invocation
} MLIR_PassContext;

typedef enum {
    MLIR_ANCHOR_MODULE = 0,
    MLIR_ANCHOR_FUNCTION
} MLIR_PassAnchor;

typedef bool (*MLIR_PassFn)(MLIR_PassContext *pc, MLIR_OpHandle anchor, void *state);

typedef struct MLIR_Pass {
    string          name;            // for timing, diagnostics, print_after
    MLIR_PassAnchor anchor;          // FUNCTION = every op with MLIR_T_FUNCTION and a body
    MLIR_PassFn     run;
    void           *state;
} MLIR_Pass;

typedef struct MLIR_PipelineOptions {
    const MLIR_TargetInfo *target;       // passed through to MLIR_PassContext.target
    bool                   verify_each;  // MLIR_VerifyOp after every pass
    bool                   timing;       // per-pass wall time via MLIR_DIAG_REMARK
    string                 print_after;  // pass name or "*"; empty = none
} MLIR_PipelineOptions;

// Runs `passes[0..n-1]` in order on `module`. Returns true if every pass
// (and verification when `verify_each` is set) succeeded. `opts` may be NULL.
bool MLIR_RunPasses(MLIR_Context *ctx, MLIR_OpHandle module,
                    const MLIR_Pass *passes, size_t n,
                    const MLIR_PipelineOptions *opts);

// Standard in-tree passes as MLIR_Pass values. Each constructor lives in its
// pass's translation unit so callers only link the passes they reference.
MLIR_Pass MLIR_PassMem2Reg(void);
MLIR_Pass MLIR_PassLoadCSE(void);
MLIR_Pass MLIR_PassArithGVN(void);
MLIR_Pass MLIR_PassDCE(void);
MLIR_Pass MLIR_PassLowerToLLVMDialect(void);
MLIR_Pass MLIR_PassLiftCfToScf(const MLIR_LiftCfToScfOptions *options);

#ifdef __cplusplus
}
#endif

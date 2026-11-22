// Upstream-only entry points (#151).
//
// These functions run upstream MLIR/LLVM (its printer, parser, pass
// pipeline, LLVM-IR translator and LLVM target machine) and are
// implemented only by mlir_api_impl_upstream.cpp. Include this header
// only in code that links the upstream backend; with the native backend,
// MLIR_LowerToLLVMDialectUpstream and MLIR_TranslateModuleTo*Upstream
// fail at link time. The native backend keeps two stubs so that the
// shared `parser` test driver links against either backend:
// MLIR_PrintOperationUpstream returns an "error: ..." string and
// MLIR_ParseTextUpstream returns MLIR_INVALID_HANDLE.
//
// They exist so callers can run upstream side by side with the in-tree
// implementations (mlir_printer.h, mlir_parser.h, mlir_passes.h) for
// reference, performance and correctness comparisons. All of them print
// diagnostics to stderr on failure.

#pragma once

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Print `op` with upstream's Operation::print.
string MLIR_PrintOperationUpstream(MLIR_Context *ctx, MLIR_OpHandle op);

// Parse a top-level MLIR module from text with upstream's parser. Returns
// MLIR_INVALID_HANDLE on failure.
MLIR_OpHandle MLIR_ParseTextUpstream(MLIR_Context *ctx, string text);

// Upstream siblings of MLIR_LowerToLLVMDialect and
// MLIR_TranslateModuleTo{LLVMIR,Wasm} (mlir_passes.h): the same contracts,
// implemented with upstream's PassManager, LLVM-IR translation and the
// LLVM WebAssembly target machine.
bool MLIR_LowerToLLVMDialectUpstream(MLIR_Context *ctx, MLIR_OpHandle module);
string MLIR_TranslateModuleToLLVMIRUpstream(MLIR_Context *ctx, MLIR_OpHandle module);
string MLIR_TranslateModuleToWasmUpstream(MLIR_Context *ctx, MLIR_OpHandle module);

#ifdef __cplusplus
}
#endif

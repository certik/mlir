#include "api/bench_lower_pipeline.h"

const char *bench_lower_stage_name(BenchLowerStage stage) {
    switch (stage) {
    case BENCH_LOWER_STAGE_CF_TO_SCF:           return "cf -> scf";
    case BENCH_LOWER_STAGE_TO_LLVM_KEEP_SCF:    return "arith/memref -> llvm (keep scf)";
    case BENCH_LOWER_STAGE_LLVM_TO_WASMSSA:     return "llvm -> wasmssa";
    case BENCH_LOWER_STAGE_WASMSSA_TO_WASMSTACK: return "wasmssa -> wasmstack";
    case BENCH_LOWER_STAGE_WASMSTACK_TO_WASM:   return "wasmstack -> wasm";
    case BENCH_LOWER_STAGE_SCF_CF_TO_LLVM:      return "scf/cf -> llvm";
    case BENCH_LOWER_STAGE_LLVM_TO_ELF:         return "llvm -> x86 (elf)";
    case BENCH_LOWER_STAGE_LLVM_TO_AARCH64:     return "llvm -> aarch64";
    case BENCH_LOWER_STAGE_AARCH64_TO_MACHO:    return "aarch64 -> macho";
    default:                                    return "?";
    }
}

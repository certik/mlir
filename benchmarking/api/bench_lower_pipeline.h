#pragma once

#if defined(__linux__) && defined(__x86_64__)
#define BENCH_LOWER_NATIVE_X64 1
#define BENCH_LOWER_HAS_NATIVE 1
#elif defined(__APPLE__) && defined(__aarch64__)
#define BENCH_LOWER_NATIVE_AARCH64 1
#define BENCH_LOWER_HAS_NATIVE 1
#else
#define BENCH_LOWER_NATIVE_X64 0
#define BENCH_LOWER_NATIVE_AARCH64 0
#define BENCH_LOWER_HAS_NATIVE 0
#endif

// Legacy alias used by existing conditionals.
#define BENCH_LOWER_PIPELINE_HAS_X64 BENCH_LOWER_NATIVE_X64

typedef enum {
    BENCH_LOWER_STAGE_CF_TO_SCF = 0,
    BENCH_LOWER_STAGE_TO_LLVM_KEEP_SCF,
    BENCH_LOWER_STAGE_LLVM_TO_WASMSSA,
    BENCH_LOWER_STAGE_WASMSSA_TO_WASMSTACK,
    BENCH_LOWER_STAGE_WASMSTACK_TO_WASM,
    BENCH_LOWER_STAGE_SCF_CF_TO_LLVM,
    BENCH_LOWER_STAGE_LLVM_TO_ELF,
    BENCH_LOWER_STAGE_LLVM_TO_AARCH64,
    BENCH_LOWER_STAGE_AARCH64_TO_MACHO,
} BenchLowerStage;

const char *bench_lower_stage_name(BenchLowerStage stage);

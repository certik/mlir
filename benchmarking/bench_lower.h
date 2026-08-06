#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <platform/platform.h>
#include <base/arena.h>
#include <base/string.h>
#include "mlir_api.h"
#include "tinyc.h"

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

typedef struct {
    double cf_to_scf_ms;
    double to_llvm_keep_scf_ms;
    double llvm_to_wasmssa_ms;
    double wasmssa_to_wasmstack_ms;
    double wasmstack_to_wasm_ms;
    size_t wasm_bytes;
} BenchLowerWasmTimings;

typedef struct {
    double scf_cf_to_llvm_ms;
#if BENCH_LOWER_NATIVE_X64
    double llvm_to_elf_ms;
#elif BENCH_LOWER_NATIVE_AARCH64
    double llvm_to_aarch64_ms;
    double aarch64_to_macho_ms;
#endif
    size_t binary_bytes;
    bool binary_ok;
} BenchLowerNativeTimings;

static inline double bench_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static inline size_t bench_n(void) {
    const char *e = getenv("BENCH_N");
    if (!e || !e[0]) return 10000;
    unsigned long v = strtoul(e, NULL, 10);
    return v ? (size_t)v : 10000;
}

bool bench_lower_load_tc(Arena *arena, MLIR_Context *ctx, const char *path,
                         Program **out_prog, MLIR_OpHandle *out_mod);
bool bench_lower_run_frontend(Arena *arena, MLIR_Context *ctx, const char *path,
                              Program **out_prog, MLIR_OpHandle *out_mod,
                              double *out_ms);
bool bench_lower_emit_exec_native(Arena *arena, const char *tc_path,
                                  const char *out_path);
bool bench_lower_run_wasm(MLIR_Context *ctx, MLIR_OpHandle module,
                          BenchLowerWasmTimings *out);
bool bench_lower_run_native(MLIR_Context *ctx, Arena *arena, Program *prog,
                            BenchLowerNativeTimings *out);
bool bench_lower_bench_file(const char *tc_path);
void bench_lower_print_wasm(const BenchLowerWasmTimings *t);
void bench_lower_print_native(const BenchLowerNativeTimings *t);

string bench_lower_construct_tc(Arena *arena, size_t n_funcs);
string bench_lower_construct_clang_c(Arena *arena, size_t n_funcs);

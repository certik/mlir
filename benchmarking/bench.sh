#!/usr/bin/env bash
# Benchmark driver: build small compiler (bench_lower) + full compiler
# (tinyc_native_opt), generate workload, run per-stage timings and/or the
# 5-way compare table (clang, clang -O0, clang -O3 -march=native,
# small compiler, full compiler).
#
# Usage:
#   bash benchmarking/bench.sh [compare|stages|all]   (default: compare)
#   BENCH_N=10000 REPEATS=3 BENCH_STAT=avg bash benchmarking/bench.sh stages
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-compare}"
: "${CC:=clang}"
: "${BENCH_N:=10000}"
: "${REPEATS:=3}"
: "${BENCH_STAT:=avg}"
: "${CLANG_O3_FLAGS:=-O3 -DNDEBUG -flto -march=native}"
: "${PLATFORM_C:=corec/platform/platform_linux.c}"
PLATFORM_OBJ="$(basename "$PLATFORM_C" .c).o"

OPT_FLAGS="-O3 -DNDEBUG -flto"
INC="-I corec -I . -I examples/tinyc -I benchmarking"
case "$(uname)" in
    Linux) GROUP_START="-Wl,--start-group"; GROUP_END="-Wl,--end-group" ;;
    *)     GROUP_START="";                  GROUP_END="" ;;
esac

WORK="$ROOT/benchmarking/build/workload"
COMPARE_DIR="$ROOT/benchmarking/build/compare"
BENCH_LOWER="$ROOT/benchmarking/bench_lower"
TINYC_OPT="$ROOT/tinyc_native_opt"

bench_build_tinyc() {
    export CC PLATFORM_C PLATFORM_OBJ
    bash examples/tinyc/build_tinyc_native_opt.sh
    echo "built $TINYC_OPT ($OPT_FLAGS)"
}

bench_build_lower() {
    local obj_dir="benchmarking/build_lower"
    rm -rf "$obj_dir"
    mkdir -p "$obj_dir"

    local -a corec_c=(
        corec/base/io.c corec/base/buddy.c corec/base/arena.c
        corec/base/scratch.c corec/base/format.c corec/base/math.c
        corec/base/string.c corec/base/strbuf.c corec/base/mem.c
        corec/base/numconv.c corec/base/assert.c corec/base/exit.c
    )
    local -a driver_c=(
        benchmarking/bench_lower.c benchmarking/bench_common.c
        benchmarking/bench_timer.c benchmarking/api/bench_lower_pipeline.c
    )
    local -a tinyc_c=(
        examples/tinyc/lex.c examples/tinyc/preprocess.c
        examples/tinyc/parse.c examples/tinyc/emit.c
    )
    local -a api_c=(
        mlir_api_impl.c mlir_op_names.c mlir_parser.c tokenizer.c op_parsers.c
    )
    local -a pipeline_c=(
        mlir_lift_cf_to_scf.c mlir_lower_to_llvm.c mlir_llvm_to_wasmssa.c
        mlir_wasmssa_to_wasmstack.c mlir_wasmstack_to_bin.c
    )
    local -a x64_c=(
        mlir_llvm_to_x64.c mlir_elf.c mlir_regalloc.c
        benchmarking/api/bench_lower_regalloc_hooks.c
    )

    compile() { $CC $OPT_FLAGS -c $INC -o "$1" "${@:2}"; }

    compile "$obj_dir/upstream_main.o" tests/upstream_main.c
    local f base
    for f in "${driver_c[@]}" "${tinyc_c[@]}" "${api_c[@]}" "${pipeline_c[@]}" "${corec_c[@]}"; do
        base="$(basename "$f" .c).o"
        compile "$obj_dir/$base" "$f"
    done

    local -a x64_objs=()
    if [ "$(uname)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
        for f in "${x64_c[@]}"; do
            base="$(basename "$f" .c).o"
            compile "$obj_dir/$base" "$f"
            x64_objs+=("$obj_dir/$base")
        done
    fi

    compile "$obj_dir/$PLATFORM_OBJ" -DPLATFORM_SKIP_ENTRY "$PLATFORM_C"

    local -a objs=("$obj_dir/upstream_main.o")
    for f in "${driver_c[@]}" "${tinyc_c[@]}" "${api_c[@]}" "${pipeline_c[@]}" "${corec_c[@]}"; do
        objs+=("$obj_dir/$(basename "$f" .c).o")
    done

    $CC $OPT_FLAGS -o "$BENCH_LOWER" \
        "${objs[@]}" "${x64_objs[@]}" "$obj_dir/$PLATFORM_OBJ" \
        $GROUP_START $GROUP_END -lm
    echo "built $BENCH_LOWER ($OPT_FLAGS)"
}

bench_generate_workload() {
    mkdir -p "$WORK"
    python3 - "$WORK" "$BENCH_N" <<'PY'
import os, sys
work, n = sys.argv[1], int(sys.argv[2])

def tc_func(i):
    num, hi = i + 1, i + 10
    return (f"void g{num}(int *x) {{\n    int i;\n    *x = 0;\n"
            f"    for (i = {num}; i <= {hi}; i = i + 1) {{\n"
            f"        *x = *x + i;\n    }}\n}}\n")

def c_func(i):
    num, hi = i + 1, i + 10
    return (f"void g{num}(int *x) {{\n    int i;\n    *x = 0;\n"
            f"    for (i = {num}; i <= {hi}; i = i + 1) {{\n"
            f"        *x = *x + i;\n    }}\n}}\n")

tc_path = os.path.join(work, "bench_lower.tc")
c_path = os.path.join(work, "workload.c")
with open(tc_path, "w") as f:
    f.write("void buddy_init(void) {}\n")
    f.write("void platform_init(int argc, char **argv, char **envp) {\n")
    f.write("    (void)argc; (void)argv; (void)envp; buddy_init(); }\n")
    for i in range(n):
        f.write(tc_func(i))
    f.write("int main() { int c = 0;\n")
    for i in range(n):
        f.write(f"    g{i+1}(&c);\n")
    f.write("    return c;\n}\n")
with open(c_path, "w") as f:
    for i in range(n):
        f.write(c_func(i))
    f.write("int main() { int c = 0;\n")
    for i in range(n):
        f.write(f"    g{i+1}(&c);\n")
    f.write("    return c;\n}\n")
PY
}

bench_run_stages() {
    export BENCH_N
    exec "$BENCH_LOWER" "$WORK/bench_lower.tc"
}

bench_run_compare() {
    mkdir -p "$COMPARE_DIR"
    export BENCH_N REPEATS BENCH_STAT CLANG_O3_FLAGS CC
    export TC="$WORK/bench_lower.tc"
    export SRC="$WORK/workload.c"
    export WORK="$COMPARE_DIR"
    export BENCH_LOWER
    export CLANG_BIN="$COMPARE_DIR/workload_clang"
    export CLANG_BIN_O0="$COMPARE_DIR/workload_o0"
    export CLANG_BIN_O3="$COMPARE_DIR/workload_o3"
    export OUR_BIN_BL="$COMPARE_DIR/workload_bench_lower"
    export OUR_BIN_TO="$COMPARE_DIR/workload_tinyc_native_opt"
    export TINYC_OPT
    export CLANG_INC="-I$ROOT/corec -I$ROOT"
    python3 "$ROOT/benchmarking/bench_compare.py"
}

case "$MODE" in
    compare|stages|all) ;;
    -h|--help)
        echo "usage: $0 [compare|stages|all]  (default: all)"
        exit 0 ;;
    *)
        echo "unknown mode: $MODE (use compare, stages, or all)" >&2
        exit 1 ;;
esac

bench_build_lower
bench_build_tinyc
bench_generate_workload

case "$MODE" in
    compare) bench_run_compare ;;
    stages)  bench_run_stages ;;
    all)
        bench_run_stages
        echo
        bench_run_compare
        ;;
esac

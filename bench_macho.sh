#!/usr/bin/env bash
# Benchmark single-TU wasm compile + macho link step across compilers.
set -eu

if [ ! -x tinyc_native_opt ]; then pixi run build_tinyc_native_opt >/dev/null; fi
if [ ! -f tinyc.wasm ]; then pixi run -e wasm build_tinyc_wasm >/dev/null; fi

# Build stage1 macho binary if missing.
mkdir -p bench_stage
if [ ! -x bench_stage/tinyc_stage1 ]; then
  bash examples/tinyc/selfhost_tinyc_macho.sh tinyc.wasm bench_stage/tinyc_stage1 bench_stage >/dev/null
fi

# Build the .wasm.o object set if missing.
ALL_SRC=(
  corec/base/io.c corec/base/buddy.c corec/base/arena.c corec/base/scratch.c
  corec/base/format.c corec/base/math.c corec/base/string.c corec/base/strbuf.c
  corec/base/mem.c corec/base/numconv.c corec/base/assert.c corec/base/exit.c
  corec-stdlib/stdlib/stdio.c corec-stdlib/stdlib/stdlib.c
  corec-stdlib/stdlib/printf.c corec-stdlib/stdlib/string_impl.c
  examples/tinyc/lex.c examples/tinyc/preprocess.c examples/tinyc/parse.c
  examples/tinyc/emit.c examples/tinyc/driver.c
  mlir_api_impl.c mlir_op_names.c mlir_lower_to_llvm.c
  mlir_translate_to_llvm_ir.c mlir_translate_to_wasm.c mlir_wasm_to_wat.c
  mlir_wasm_to_macho.c mlir_llvm_to_wasmssa.c mlir_wasmssa_to_wasmstack.c
  mlir_wasmstack_to_bin.c mlir_wasm_link.c tokenizer.c mlir_parser.c
  op_parsers.c mlir_classic_printer.c mlir_generic_printer.c
  mlir_lift_cf_to_scf.c corec/platform/platform_wasm.c
)
INCLUDES=(-I corec -I corec-stdlib/stdlib -I .)
if [ ! -d bench_objs ] || [ "$(ls bench_objs/*.wasm.o 2>/dev/null | wc -l | tr -d ' ')" -lt 40 ]; then
  rm -rf bench_objs && mkdir -p bench_objs
  for src in "${ALL_SRC[@]}"; do
    base=$(echo "$src" | tr '/' '_')
    ./tinyc_native_opt --emit=wasm --lowering=native "${INCLUDES[@]}" \
        -o "bench_objs/${base}.wasm.o" "$src" >/dev/null
  done
  cp tinyc_wasm_vararg.wasm.o bench_objs/
fi

OBJS=(bench_objs/*.wasm.o)
TARGET="${1:-mlir_wasm_to_macho.c}"

runtime() {
    local label="$1"; shift
    local t
    t=$( { time "$@" >/dev/null 2>&1 ; } 2>&1 | grep real | awk '{print $2}')
    printf '  %-32s %s\n' "$label" "$t"
}

echo "=== Single TU compile ($TARGET) ==="
runtime "tinyc_native_opt (clang -O3)" ./tinyc_native_opt \
    --emit=wasm --lowering=native "${INCLUDES[@]}" -o /tmp/bench_out.wasm.o "$TARGET"
runtime "tinyc.wasm via wasmtime" wasmtime --dir . tinyc.wasm \
    --emit=wasm --lowering=native "${INCLUDES[@]}" -o /tmp/bench_out.wasm.o "$TARGET"
runtime "tinyc-compiled macho" ./bench_stage/tinyc_stage1 \
    --emit=wasm --lowering=native "${INCLUDES[@]}" -o /tmp/bench_out.wasm.o "$TARGET"

echo ""
echo "=== Link step (40 objs -> macho) ==="
runtime "tinyc_native_opt (clang -O3)" ./tinyc_native_opt --link \
    --emit=macho --export=_start -o /tmp/bench_out_macho "${OBJS[@]}"
runtime "tinyc.wasm via wasmtime" wasmtime --dir . tinyc.wasm --link \
    --emit=macho --export=_start -o /tmp/bench_out_macho "${OBJS[@]}"
runtime "tinyc-compiled macho" ./bench_stage/tinyc_stage1 --link \
    --emit=macho --export=_start -o /tmp/bench_out_macho "${OBJS[@]}"

echo ""
echo "Output binary size: $(ls -la /tmp/bench_out_macho | awk '{print $5}') bytes"

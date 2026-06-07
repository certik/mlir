#!/usr/bin/env bash
#
# Self-host the tinyC compiler: use an existing `tinyc.wasm` (under
# wasmtime) to recompile every source file that makes up the tinyc
# wasm32-wasi build into `.wasm.o` objects, then use that same
# `tinyc.wasm` to link them into a fresh `tinyc.wasm` binary.
#
# Usage:
#   selfhost_tinyc_wasm.sh <INPUT_TINYC_WASM> <OUTPUT_TINYC_WASM> <STAGE_DIR>
#
# The set of source files mirrors `build_tinyc_wasm.sh` exactly so the
# stage-2 and stage-3 outputs can be compared byte-for-byte to verify
# self-hosting reproducibility.

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 INPUT_TINYC_WASM OUTPUT_TINYC_WASM STAGE_DIR" >&2
    exit 2
fi

INPUT_WASM="$1"
OUTPUT_WASM="$2"
STAGE_DIR="$3"

if [ ! -f "$INPUT_WASM" ]; then
    echo "error: input compiler not found: $INPUT_WASM" >&2
    exit 1
fi

mkdir -p "$STAGE_DIR"

# Source set mirrors `build_tinyc_wasm.sh`. Keep these in sync.
COREC_C_FILES=(
    corec/base/io.c
    corec/base/buddy.c
    corec/base/arena.c
    corec/base/scratch.c
    corec/base/format.c
    corec/base/math.c
    corec/base/string.c
    corec/base/strbuf.c
    corec/base/mem.c
    corec/base/numconv.c
    corec/base/assert.c
    corec/base/exit.c
)
COREC_STDLIB_C_FILES=(
    corec-stdlib/stdlib/stdio.c
    corec-stdlib/stdlib/stdlib.c
    corec-stdlib/stdlib/printf.c
    corec-stdlib/stdlib/string_impl.c
)
TINYC_C_FILES=(
    examples/tinyc/lex.c
    examples/tinyc/preprocess.c
    examples/tinyc/parse.c
    examples/tinyc/emit.c
    examples/tinyc/driver.c
)
NATIVE_C_FILES=(
    mlir_api_impl.c
    mlir_op_names.c
    mlir_lower_to_llvm.c
    mlir_translate_to_llvm_ir.c
    mlir_translate_to_wasm.c
    mlir_wasm_to_wat.c
    mlir_wasm_to_macho.c
    mlir_llvm_to_wasmssa.c
    mlir_wasmssa_to_llvm.c
    mlir_wasmssa_to_wasmstack.c
    mlir_wasmstack_to_bin.c
    mlir_wasm_link.c
    mlir_wasm_to_wasmstack.c
    mlir_wasmstack_to_wasmssa.c
    mlir_llvm_mem2reg.c
    mlir_llvm_load_cse.c
    mlir_llvm_arith_gvn.c
    mlir_llvm_dce.c
    mlir_llvm_to_aarch64.c
    mlir_aarch64_to_macho.c
    tokenizer.c
    mlir_parser.c
    op_parsers.c
    mlir_classic_printer.c
    mlir_generic_printer.c
    mlir_lift_cf_to_scf.c
)
PLATFORM_C_FILES=(
    corec/platform/platform_wasm.c
)

ALL_SOURCES=(
    "${COREC_C_FILES[@]}"
    "${COREC_STDLIB_C_FILES[@]}"
    "${TINYC_C_FILES[@]}"
    "${NATIVE_C_FILES[@]}"
    "${PLATFORM_C_FILES[@]}"
)

INCLUDES=(-I corec -I corec-stdlib/stdlib -I .)

# A tinyC-compiled wasm executable needs a tiny clang-built runtime
# shim with the `tinyc_va_arg_*` helpers (tinyC lowers va_arg to
# direct calls into them). `_start` is provided by tinyC-compiled
# `corec/platform/platform_wasm.c`, so unlike the clang-built
# `tinyc.wasm` we do NOT pull in `start_wasm.wasm.o` here (its
# hand-written stub references the clang-only `__original_main`).
VARARG_OBJ=tinyc_wasm_vararg.wasm.o
for o in "$VARARG_OBJ"; do
    if [ ! -f "$o" ]; then
        echo "error: required object $o not found; run \`pixi run build_tinyc_wasm\` first" >&2
        exit 1
    fi
done

OBJS=()
for src in "${ALL_SOURCES[@]}"; do
    base="$(echo "$src" | tr '/' '_')"
    obj="$STAGE_DIR/${base}.wasm.o"
    printf '[tinyc.wasm] %s -> %s\n' "$src" "$obj"
    wasmtime --dir . --dir "$STAGE_DIR" "$INPUT_WASM" \
        --emit=wasm --lowering=native \
        "${INCLUDES[@]}" \
        -o "$obj" "$src"
    OBJS+=("$obj")
done

printf '[tinyc.wasm] --link -> %s\n' "$OUTPUT_WASM"
wasmtime --dir . --dir "$STAGE_DIR" "$INPUT_WASM" --link \
    --export=_start \
    -o "$OUTPUT_WASM" "${OBJS[@]}" "$VARARG_OBJ"

ls -l "$OUTPUT_WASM"

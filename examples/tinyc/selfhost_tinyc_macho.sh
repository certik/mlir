#!/usr/bin/env bash
#
# Self-host the tinyC compiler into a native Mach-O binary on macOS:
# use an existing tinyC compiler (either a `.wasm` driven by wasmtime
# or a previously-bootstrapped native Mach-O `tinyc` binary) to
# recompile every source file that makes up the tinyc wasm32-wasi
# build into `.wasm.o` objects, then use that same compiler to
# `--link --emit=macho` them into a runnable native Mach-O executable.
#
# Usage:
#   selfhost_tinyc_macho.sh <INPUT_TINYC> <OUTPUT_TINYC_MACHO> <STAGE_DIR>
#
# `<INPUT_TINYC>` ending in `.wasm` is invoked via `wasmtime --dir .`.
# Anything else is invoked directly (must be an executable native
# Mach-O binary built by a prior stage).
#
# The source set mirrors `selfhost_tinyc_wasm.py` exactly so stage-2
# and stage-3 binaries can be compared bit-for-bit to verify
# self-hosting reproducibility.
#
# Wired into pixi via the macOS-arm64 tasks
# `selfhost_tinyc_macho_stage{1,2,3}` and `verify_tinyc_macho_selfhost`
# (under `pixi run -e wasm`).

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: $0 INPUT_TINYC OUTPUT_TINYC_MACHO STAGE_DIR" >&2
    exit 2
fi

INPUT_TINYC="$1"
OUTPUT_MACHO="$2"
STAGE_DIR="$3"

if [ ! -e "$INPUT_TINYC" ]; then
    echo "error: input compiler not found: $INPUT_TINYC" >&2
    exit 1
fi

mkdir -p "$STAGE_DIR"

# Invocation prefix: `.wasm` inputs run under wasmtime; anything else
# is invoked directly.
case "$INPUT_TINYC" in
    *.wasm)
        TINYC_INVOKE=(wasmtime --dir . --dir "$STAGE_DIR" "$INPUT_TINYC")
        ;;
    *)
        TINYC_INVOKE=("$INPUT_TINYC")
        ;;
esac

# Source set mirrors `selfhost_tinyc_wasm.py`.  Keep these in sync.
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
    mlir_regalloc.c
    mlir_llvm_to_aarch64.c
    mlir_aarch64_asm.c
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

# tinyC lowers va_arg inline (the portable 8-byte cursor model), so the
# self-hosted module needs no external support objects at link time.

OBJS=()
for src in "${ALL_SOURCES[@]}"; do
    base="$(echo "$src" | tr '/' '_')"
    obj="$STAGE_DIR/${base}.wasm.o"
    printf '[selfhost-macho] %s -> %s\n' "$src" "$obj"
    "${TINYC_INVOKE[@]}" \
        --emit=wasm --lowering=native \
        "${INCLUDES[@]}" \
        -o "$obj" "$src"
    OBJS+=("$obj")
done

printf '[selfhost-macho] --link --emit=macho -> %s\n' "$OUTPUT_MACHO"
"${TINYC_INVOKE[@]}" --link --emit=macho \
    --export=_start \
    -o "$OUTPUT_MACHO" "${OBJS[@]}"

chmod +x "$OUTPUT_MACHO"
ls -l "$OUTPUT_MACHO"

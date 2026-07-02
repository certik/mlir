#!/usr/bin/env bash
#
# Self-host the tinyC compiler into a native Linux ELF binary:
# use an existing tinyC compiler (either `tinyc.wasm` driven by wasmtime
# or a previously-bootstrapped native tinyC binary) to recompile every
# source file that makes up the native tinyC build into per-source
# `.wasm.o` objects, link them with `tinyc --link` into a single wasm
# module, and then lift that linked module back through
#   wasm -> wasmstack -> wasmssa -> llvm -> x86_64 -> ELF
# via `tinyc --from-wasm --emit=elf`.
#
# Usage:
#   selfhost_tinyc_elf.sh <INPUT_TINYC> <OUTPUT_TINYC_ELF> <STAGE_DIR>
#
# `<INPUT_TINYC>` ending in `.wasm` is invoked via `wasmtime --dir .`.
# Anything else is invoked directly (must be an executable native tinyC
# binary built by a prior stage).

set -euo pipefail

# Lifted wasm code uses large shadow-stack frames; default 8 MiB stack can
# overflow when compiling bigger sources (e.g. mlir_op_names.c) after many
# prior compiles in this loop.
ulimit -s unlimited 2>/dev/null || true

if [ "$#" -ne 3 ]; then
    echo "usage: $0 INPUT_TINYC OUTPUT_TINYC_ELF STAGE_DIR" >&2
    exit 2
fi

if [ "$(uname)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "error: this self-host path is Linux/x86_64 only." >&2
    exit 1
fi

INPUT_TINYC="$1"
OUTPUT_ELF="$2"
STAGE_DIR="$3"

if [ ! -e "$INPUT_TINYC" ]; then
    echo "error: input compiler not found: $INPUT_TINYC" >&2
    exit 1
fi

mkdir -p "$STAGE_DIR"

case "$INPUT_TINYC" in
    *.wasm)
        TINYC_INVOKE=(wasmtime --dir . --dir "$STAGE_DIR" "$INPUT_TINYC")
        ;;
    *)
        TINYC_INVOKE=("$INPUT_TINYC")
        ;;
esac

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
    mlir_wasmssa_to_wasmstack.c
    mlir_wasmstack_to_bin.c
    mlir_wasm_link.c
    mlir_wasm_to_wasmstack.c
    mlir_wasmstack_to_wasmssa.c
    mlir_wasmssa_to_llvm.c
    mlir_llvm_mem2reg.c
    mlir_llvm_load_cse.c
    mlir_llvm_arith_gvn.c
    mlir_llvm_dce.c
    mlir_regalloc.c
    mlir_llvm_to_aarch64.c
    mlir_aarch64_asm.c
    mlir_aarch64_to_macho.c
    mlir_llvm_to_x64.c
    mlir_elf.c
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

OBJS=()
for src in "${ALL_SOURCES[@]}"; do
    base="$(echo "$src" | tr '/' '_')"
    obj="$STAGE_DIR/${base}.wasm.o"
    printf '[selfhost-elf] %s -> %s\n' "$src" "$obj"
    "${TINYC_INVOKE[@]}" \
        --emit=wasm --lowering=native \
        -DTINYC_ENABLE_ELF=1 \
        "${INCLUDES[@]}" \
        -o "$obj" "$src"
    OBJS+=("$obj")
done

LINKED_WASM="$STAGE_DIR/linked.wasm"
printf '[selfhost-elf] tinyc --link -> %s\n' "$LINKED_WASM"
"${TINYC_INVOKE[@]}" --link --export=_start \
    -o "$LINKED_WASM" \
    "${OBJS[@]}"

printf '[selfhost-elf] --from-wasm --emit=elf -> %s\n' "$OUTPUT_ELF"
"${TINYC_INVOKE[@]}" --from-wasm "$LINKED_WASM" \
    --emit=elf \
    --host-platform corec/platform/platform_linux.c \
    --wasi-adapter corec/wasm/wasi_adapter.c \
    -I corec -I . \
    -o "$OUTPUT_ELF"

chmod +x "$OUTPUT_ELF"
ls -l "$OUTPUT_ELF"

#!/usr/bin/env bash
#
# Self-host the tinyC compiler into a native Mach-O binary via the
# wmir backend: use an existing tinyC compiler (either `tinyc.wasm`
# driven by wasmtime or a previously-bootstrapped native Mach-O
# `tinyc` binary) to recompile every source file that makes up the
# tinyc wasm32-wasi build into per-source `.wasm.o` objects, link
# them with `wasm-ld` into a single linked module, and then translate
# that linked module to a native Mach-O ARM64 binary via the new
#   wasm -> wasmstack -> wasmssa -> wmir -> aarch64 -> Mach-O
# pipeline (`tinyc --from-wasm <linked.wasm> --emit=macho
# --macho-backend=wmir`).
#
# Usage:
#   selfhost_tinyc_macho_wmir.sh <INPUT_TINYC> <OUTPUT_TINYC_MACHO> <STAGE_DIR>
#
# `<INPUT_TINYC>` ending in `.wasm` is invoked via `wasmtime --dir .`.
# Anything else is invoked directly (must be an executable native
# Mach-O binary built by a prior stage).
#
# The source set mirrors `selfhost_tinyc_wasm.sh` exactly so stage-2
# and stage-3 binaries can be compared bit-for-bit to verify
# self-hosting reproducibility through the wmir backend.

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

case "$INPUT_TINYC" in
    *.wasm)
        TINYC_INVOKE=(wasmtime --dir . --dir "$STAGE_DIR" "$INPUT_TINYC")
        ;;
    *)
        TINYC_INVOKE=("$INPUT_TINYC")
        ;;
esac

# Source set mirrors `selfhost_tinyc_wasm.sh`. Keep these in sync.
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
    mlir_wasmssa_to_wmir.c
    mlir_wmir_regalloc.c
    mlir_wmir_to_aarch64.c
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
# direct calls into them). Mirrors `selfhost_tinyc_wasm.sh`.
VARARG_OBJ=tinyc_wasm_vararg.wasm.o
if [ ! -f "$VARARG_OBJ" ]; then
    echo "error: required object $VARARG_OBJ not found; run \`pixi run build_tinyc_wasm\` first" >&2
    exit 1
fi

OBJS=()
for src in "${ALL_SOURCES[@]}"; do
    base="$(echo "$src" | tr '/' '_')"
    obj="$STAGE_DIR/${base}.wasm.o"
    printf '[selfhost-macho-wmir] %s -> %s\n' "$src" "$obj"
    "${TINYC_INVOKE[@]}" \
        --emit=wasm --lowering=native \
        "${INCLUDES[@]}" \
        -o "$obj" "$src"
    OBJS+=("$obj")
done

# wasm-ld step: produce a single linked wasm module with `_start`
# exported. The wmir backend's --from-wasm path consumes this linked
# module and lowers it to Mach-O directly.
LINKED_WASM="$STAGE_DIR/linked.wasm"
printf '[selfhost-macho-wmir] wasm-ld -> %s\n' "$LINKED_WASM"
wasm-ld --no-entry --export=_start \
    -o "$LINKED_WASM" \
    "${OBJS[@]}" "$VARARG_OBJ"

# Final stage: lift wasm back into wasmssa, then run the new wmir
# pipeline all the way to a signed ad-hoc Mach-O ARM64 binary.
printf '[selfhost-macho-wmir] --from-wasm --emit=macho --macho-backend=wmir -> %s\n' "$OUTPUT_MACHO"
"${TINYC_INVOKE[@]}" --from-wasm "$LINKED_WASM" \
    --emit=macho --macho-backend=wmir \
    -o "$OUTPUT_MACHO"

chmod +x "$OUTPUT_MACHO"
ls -l "$OUTPUT_MACHO"

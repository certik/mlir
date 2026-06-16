#!/usr/bin/env bash
#
# Self-host the tinyC compiler into a native Mach-O binary via the
# llvm backend: use an existing tinyC compiler (either `tinyc.wasm`
# driven by wasmtime or a previously-bootstrapped native Mach-O
# `tinyc` binary) to recompile every source file that makes up the
# tinyc wasm32-wasi build into per-source `.wasm.o` objects, link
# them with `tinyc --link` into a single linked module, and then
# translate that linked module to a native Mach-O ARM64 binary via
# the new
#   wasm -> wasmstack -> wasmssa -> llvm -> aarch64 -> Mach-O
# pipeline (`tinyc --from-wasm <linked.wasm> --emit=macho
# --macho-backend=llvm`).
#
# Note: we deliberately use `tinyc --link` rather than `wasm-ld`
# here. wasm-ld silently miscompiles the resulting binary in the
# linker pass — the produced `_start` faults on dereferences of
# pointers loaded from the data segment (e.g. emit_line_directive
# reads junk for pp->cur_file.str). tinyc's own linker produces
# byte-identical output to the canonical wasm selfhost.
#
# Usage:
#   selfhost_tinyc_macho_llvm.sh <INPUT_TINYC> <OUTPUT_TINYC_MACHO> <STAGE_DIR>
#
# `<INPUT_TINYC>` ending in `.wasm` is invoked via `wasmtime --dir .`.
# Anything else is invoked directly (must be an executable native
# Mach-O binary built by a prior stage).
#
# The source set mirrors `selfhost_tinyc_wasm.py` exactly so stage-2
# and stage-3 binaries can be compared bit-for-bit to verify
# self-hosting reproducibility through the llvm backend.

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

# Source set mirrors `selfhost_tinyc_wasm.py`. Keep these in sync.
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
    mlir_llvm_mem2reg.c mlir_llvm_load_cse.c mlir_llvm_arith_gvn.c mlir_llvm_dce.c
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

# A tinyC-compiled wasm executable needs a tiny clang-built runtime
# shim with the `tinyc_va_arg_*` helpers (tinyC lowers va_arg to
# direct calls into them). Mirrors `selfhost_tinyc_wasm.py`.
VARARG_OBJ=tinyc_wasm_vararg.wasm.o
if [ ! -f "$VARARG_OBJ" ]; then
    echo "error: required object $VARARG_OBJ not found; run \`pixi run build_tinyc_wasm\` first" >&2
    exit 1
fi

OBJS=()
for src in "${ALL_SOURCES[@]}"; do
    base="$(echo "$src" | tr '/' '_')"
    obj="$STAGE_DIR/${base}.wasm.o"
    printf '[selfhost-macho-llvm] %s -> %s\n' "$src" "$obj"
    "${TINYC_INVOKE[@]}" \
        --emit=wasm --lowering=native \
        "${INCLUDES[@]}" \
        -o "$obj" "$src"
    OBJS+=("$obj")
done

# tinyc --link step: produce a single linked wasm module with
# `_start` exported. The llvm backend's --from-wasm path consumes
# this linked module and lowers it to Mach-O directly.
#
# We use `tinyc --link` (not `wasm-ld`) because wasm-ld miscompiles
# our `.wasm.o` set: the resulting `_start` faults on the first
# dereference inside `emit_line_directive_if_needed` (likely a
# relocation or data-segment layout mismatch). tinyc's own linker
# produces a working binary byte-identical to `selfhost_tinyc_wasm.py`.
LINKED_WASM="$STAGE_DIR/linked.wasm"
printf '[selfhost-macho-llvm] tinyc --link -> %s\n' "$LINKED_WASM"
"${TINYC_INVOKE[@]}" --link --export=_start \
    -o "$LINKED_WASM" \
    "${OBJS[@]}" "$VARARG_OBJ"

# Final stage: lift wasm back into wasmssa, then run the new llvm
# pipeline all the way to a signed ad-hoc Mach-O ARM64 binary. The
# WASI adapters call corec's platform_macos.c (spliced in via
# --host-platform) for the actual OS I/O + exit.
printf '[selfhost-macho-llvm] --from-wasm --emit=macho --macho-backend=llvm -> %s\n' "$OUTPUT_MACHO"
"${TINYC_INVOKE[@]}" --from-wasm "$LINKED_WASM" \
    --emit=macho --macho-backend=llvm \
    --host-platform corec/platform/platform_macos.c \
    -I corec -I . \
    -o "$OUTPUT_MACHO"

chmod +x "$OUTPUT_MACHO"
ls -l "$OUTPUT_MACHO"

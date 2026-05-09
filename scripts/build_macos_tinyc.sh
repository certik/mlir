#!/usr/bin/env bash
# Build the mlir parser binary by compiling every C source file with the
# tinyC compiler instead of clang's C front-end.
#
# Pipeline per source:
#   1. clang -E      preprocess (no system headers, corec/corec-stdlib only)
#   2. tinyc         lower the preprocessed unit to LLVM IR (.ll)
#
# Final link:
#   clang -nostdlib -fno-builtin -lSystem -Wl,-e,__start <all .ll>
#
# Mirrors the `build_macos` task except for swapping clang's frontend with
# tinyc. Output binary: ./parser_tinyc.
#
# Requirements: ./tinyc (built by `pixi run -e upstream build_tinyc_upstream`).
set -euo pipefail

TINYC="${TINYC:-./tinyc}"
if [ ! -x "$TINYC" ]; then
    echo "error: tinyC binary not found at $TINYC. Run 'pixi run -e upstream build_tinyc_upstream' first." >&2
    exit 1
fi

OUT="${OUT:-build_tinyc}"
BIN="${BIN:-parser_tinyc}"
mkdir -p "$OUT"

INCLUDES=(-I corec -I corec-stdlib/stdlib -I .)
CPP_FLAGS=(-E -P -nostdinc -fno-builtin -DNDEBUG "${INCLUDES[@]}")

SOURCES=(
    parser.c
    tokenizer.c
    mlir_parser.c
    mlir_classic_printer.c
    mlir_generic_printer.c
    op_parsers.c
    mlir_api_impl.c
    mlir_op_names.c
    corec-stdlib/stdlib/stdio.c
    corec-stdlib/stdlib/stdlib.c
    corec-stdlib/stdlib/printf.c
    corec-stdlib/stdlib/string_impl.c
    corec/base/io.c
    corec/base/buddy.c
    corec/base/arena.c
    corec/base/scratch.c
    corec/base/format.c
    corec/base/math.c
    corec/base/string.c
    corec/base/mem.c
    corec/base/numconv.c
    corec/base/assert.c
    corec/base/exit.c
    corec/platform/platform_macos.c
)

LL_FILES=()
for src in "${SOURCES[@]}"; do
    base="$(basename "$src")"
    ipath="$OUT/${base}.i"
    lpath="$OUT/${base}.ll"
    printf '[clang -E] %s\n' "$src"
    clang "${CPP_FLAGS[@]}" "$src" -o "$ipath"
    printf '[tinyc   ] %s -> %s\n' "$ipath" "$lpath"
    "$TINYC" --emit=llvm -o "$lpath" "$ipath"
    LL_FILES+=("$lpath")
done

printf '[clang   ] link -> %s\n' "$BIN"
# Compile examples/tinyc/runtime.c with clang (it intentionally uses system
# stdio.h for the print* helpers and stdarg.h for the va_arg helpers — see
# corec/scripts/build_macos_tinyc.sh for the same pattern). Only the
# va_arg helpers are referenced by the mlir parser bootstrap.
clang -nostdlib -fno-builtin -o "$BIN" "${LL_FILES[@]}" examples/tinyc/runtime.c -lSystem -Wl,-e,__start

printf 'Built %s via tinyC.\n' "$BIN"

#!/usr/bin/env bash
# Build the mlir parser binary by compiling every C source file with the
# tinyC compiler instead of clang's C front-end.
#
# Pipeline per source:
#   1. tinyc         preprocess + lower the unit to LLVM IR (.ll). tinyc
#                    has its own self-contained C preprocessor (see
#                    examples/tinyc/preprocess.c) so no external cpp is
#                    required.
#
# Final link:
#   clang -nostdlib -fno-builtin <all .ll>
#
# Mirrors the `build_linux` task except for swapping clang's frontend with
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
DEFINES=(-DNDEBUG)

SOURCES=(
    parser.c
    tokenizer.c
    mlir_parser.c
    mlir_classic_printer.c
    mlir_generic_printer.c
    op_parsers.c
    mlir_api_impl.c
    mlir_op_names.c
    mlir_lift_cf_to_scf.c
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
    corec/base/strbuf.c
    corec/base/mem.c
    corec/base/numconv.c
    corec/base/assert.c
    corec/base/exit.c
)

# platform_linux.c uses GCC/Clang-specific features (__attribute__((weak)),
# inline assembly for raw syscalls) that tinyc doesn't support, so compile
# it directly with clang and link the resulting .o.
LL_FILES=()
for src in "${SOURCES[@]}"; do
    base="$(basename "$src")"
    lpath="$OUT/${base}.ll"
    printf '[tinyc   ] %s -> %s\n' "$src" "$lpath"
    "$TINYC" --emit=llvm "${INCLUDES[@]}" "${DEFINES[@]}" -o "$lpath" "$src"
    LL_FILES+=("$lpath")
done

PLATFORM_OBJ="$OUT/platform_linux.o"
printf '[clang   ] %s -> %s\n' "corec/platform/platform_linux.c" "$PLATFORM_OBJ"
clang -c -nostdlib -fno-builtin -I corec -I corec-stdlib/stdlib -I . -DNDEBUG \
    -o "$PLATFORM_OBJ" corec/platform/platform_linux.c

printf '[clang   ] link -> %s\n' "$BIN"
clang -nostdlib -fno-builtin -Wl,--gc-sections -o "$BIN" \
    "${LL_FILES[@]}" "$PLATFORM_OBJ" examples/tinyc/tinyc_wasm_vararg.c

printf 'Built %s via tinyC.\n' "$BIN"

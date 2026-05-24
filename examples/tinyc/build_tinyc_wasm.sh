#!/usr/bin/env bash
set -e

# Build the tinyC compiler itself as a wasm32-wasi module. Output is
# tinyc.wasm — a self-contained `.wasm` that, given a tinyC source on
# its argv (via the WASI virtual FS), produces a wasm32 object on stdout
# or `-o <path>`. Combined with `--link`, the same binary also acts as
# the linker. This is what the browser demo loads.
#
# Output: ./tinyc.wasm and the prebuilt runtime/start objects used by
# `tinyc --link`.

COREC_C_FILES="corec/base/io.c corec/base/buddy.c corec/base/arena.c corec/base/scratch.c corec/base/format.c corec/base/math.c corec/base/string.c corec/base/strbuf.c corec/base/mem.c corec/base/numconv.c corec/base/assert.c corec/base/exit.c"
COREC_STDLIB_C_FILES="corec-stdlib/stdlib/stdio.c corec-stdlib/stdlib/stdlib.c corec-stdlib/stdlib/printf.c corec-stdlib/stdlib/string_impl.c"
TINYC_C_FILES="examples/tinyc/lex.c examples/tinyc/preprocess.c examples/tinyc/parse.c examples/tinyc/emit.c examples/tinyc/driver.c"
NATIVE_C_FILES="mlir_api_impl.c mlir_op_names.c mlir_lower_to_llvm.c mlir_translate_to_llvm_ir.c mlir_translate_to_wasm.c mlir_wasm_to_wat.c mlir_llvm_to_wasmssa.c mlir_wasmssa_to_wasmstack.c mlir_wasmstack_to_bin.c mlir_wasm_link.c tokenizer.c mlir_parser.c op_parsers.c mlir_classic_printer.c mlir_generic_printer.c mlir_lift_cf_to_scf.c"

clang \
    --target=wasm32-wasi \
    -Os \
    -Wl,-z,stack-size=4194304 \
    -nostdlib \
    -nostdinc \
    -fno-builtin \
    -I corec \
    -I corec-stdlib/stdlib \
    -I . \
    -Wl,--no-entry \
    -Wl,--export=_start \
    -Wl,--export=wasm_buddy_alloc \
    -Wl,--export=wasm_buddy_free \
    -Wl,--initial-memory=33554432 \
    -o tinyc.wasm \
    $COREC_C_FILES \
    $COREC_STDLIB_C_FILES \
    $TINYC_C_FILES \
    $NATIVE_C_FILES \
    corec/platform/platform_wasm.c

# Pre-build the runtime + start objects we ship alongside tinyc.wasm so
# the browser demo can hand them to `tinyc --link` without invoking
# clang in the browser.
clang --target=wasm32-wasi -O2 -nostdinc -fno-builtin \
    -I corec -I corec-stdlib/stdlib \
    -c -o runtime_wasm.wasm.o examples/tinyc/runtime_wasm.c

clang --target=wasm32 -O2 -nostdlib -fno-builtin \
    -c -o start_wasm.wasm.o examples/tinyc/start_wasm.s

# Minimal `tinyc_va_arg_*` shim used by tinyC-compiled wasm binaries
# (selfhost stage 2+, or any program tinyC compiles that uses varargs
# in a non-browser context). Kept separate from runtime_wasm.wasm.o so
# it can be linked alongside a tinyC-compiled corec-stdlib without
# duplicating printf/strlen/etc.
clang --target=wasm32-wasi -O2 -fno-builtin \
    -c -o tinyc_wasm_vararg.wasm.o examples/tinyc/tinyc_wasm_vararg.c

ls -l tinyc.wasm runtime_wasm.wasm.o start_wasm.wasm.o tinyc_wasm_vararg.wasm.o

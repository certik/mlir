#!/usr/bin/env bash
set -e

# Build the tinyC example compiler against the NATIVE MLIR backend
# (mlir_api_impl.c) instead of upstream LLVM/MLIR. The native backend
# gets its lowering + LLVM-IR translation from mlir_lower_to_llvm.c and
# mlir_translate_to_llvm_ir.c, both written against the public
# mlir_api.h surface.
#
# Output binary: ./tinyc_native.

case "$(uname)" in
    Linux) GROUP_START="-Wl,--start-group"; GROUP_END="-Wl,--end-group" ;;
    *)     GROUP_START="";                  GROUP_END="" ;;
esac

COREC_C_FILES="corec/base/io.c corec/base/buddy.c corec/base/arena.c corec/base/scratch.c corec/base/format.c corec/base/math.c corec/base/string.c corec/base/mem.c corec/base/numconv.c corec/base/assert.c corec/base/exit.c"
TINYC_C_FILES="examples/tinyc/lex.c examples/tinyc/preprocess.c examples/tinyc/parse.c examples/tinyc/emit.c examples/tinyc/driver.c"
NATIVE_C_FILES="mlir_api_impl.c mlir_op_names.c mlir_lower_to_llvm.c mlir_translate_to_llvm_ir.c mlir_translate_to_wasm.c mlir_wasm_to_wat.c mlir_llvm_to_wasmssa.c mlir_wasmssa_to_wasmstack.c mlir_wasmstack_to_bin.c mlir_wasm_link.c tokenizer.c mlir_parser.c op_parsers.c mlir_classic_printer.c mlir_generic_printer.c mlir_lift_cf_to_scf.c"

$CC -c -g -I corec -I . $COREC_C_FILES $TINYC_C_FILES $NATIVE_C_FILES tests/upstream_main.c
$CC -c -g -I corec -I . -DPLATFORM_SKIP_ENTRY $PLATFORM_C

$CC -g -o tinyc_native \
    upstream_main.o lex.o preprocess.o parse.o emit.o driver.o \
    mlir_api_impl.o mlir_op_names.o mlir_lower_to_llvm.o mlir_translate_to_llvm_ir.o mlir_translate_to_wasm.o mlir_wasm_to_wat.o mlir_llvm_to_wasmssa.o mlir_wasmssa_to_wasmstack.o mlir_wasmstack_to_bin.o mlir_wasm_link.o \
    tokenizer.o mlir_parser.o op_parsers.o mlir_classic_printer.o mlir_generic_printer.o mlir_lift_cf_to_scf.o \
    io.o buddy.o arena.o scratch.o format.o math.o string.o mem.o numconv.o assert.o exit.o $PLATFORM_OBJ \
    $GROUP_START $GROUP_END -lm

#!/usr/bin/env bash
set -e

LLVM_LIBS=$("$CONDA_PREFIX/bin/llvm-config" --link-static --libs support core analysis transformutils frontendopenmp)
SYS_LIBS="-lpthread -ldl -lm $CONDA_PREFIX/lib/libz.a -lzstd"
# GNU ld requires --start-group/--end-group to resolve cyclic dependencies
# between static archives. Apple's ld64 resolves these without help.
case "$(uname)" in
    Linux) GROUP_START="-Wl,--start-group"; GROUP_END="-Wl,--end-group" ;;
    *)     GROUP_START="";                  GROUP_END="" ;;
esac
# Link MLIR statically: enumerate every libMLIR*.a archive shipped by
# conda-forge. This pulls in all dialects (Func, Arith, MemRef, SCF, CF, ...)
# the upstream backend registers, and avoids dylib/so version-skew or rpath
# issues at runtime.
MLIR_LIBS=$(ls "$CONDA_PREFIX"/lib/libMLIR*.a 2>/dev/null)
if [ -z "$MLIR_LIBS" ]; then
    echo "error: could not find any libMLIR*.a static archives in $CONDA_PREFIX/lib" >&2
    exit 1
fi

COREC_C_FILES="corec/base/io.c corec/base/buddy.c corec/base/arena.c corec/base/scratch.c corec/base/format.c corec/base/math.c corec/base/string.c corec/base/mem.c corec/base/numconv.c corec/base/assert.c corec/base/exit.c"
PROJ_C_FILES="parser.c tokenizer.c mlir_parser.c mlir_classic_printer.c mlir_generic_printer.c op_parsers.c mlir_op_names.c"

$CC -c -g -I corec -I . $COREC_C_FILES $PROJ_C_FILES tests/upstream_main.c
$CC -c -g -I corec -I . -DPLATFORM_SKIP_ENTRY $PLATFORM_C
$CXX -c -std=c++17 -fno-rtti -g -I corec -I . -I "$CONDA_PREFIX/include" mlir_api_impl_upstream.cpp
$CXX -g -o parser_upstream \
    upstream_main.o parser.o tokenizer.o mlir_parser.o mlir_classic_printer.o mlir_generic_printer.o \
    op_parsers.o mlir_op_names.o mlir_api_impl_upstream.o \
    io.o buddy.o arena.o scratch.o format.o math.o string.o mem.o numconv.o assert.o exit.o $PLATFORM_OBJ \
    -L "$CONDA_PREFIX/lib" $GROUP_START $MLIR_LIBS $LLVM_LIBS $GROUP_END $SYS_LIBS \
    -Wl,-rpath,"$CONDA_PREFIX/lib"

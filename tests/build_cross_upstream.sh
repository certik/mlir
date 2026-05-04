#!/usr/bin/env bash
set -e

LLVM_LIBS=$("$CONDA_PREFIX/bin/llvm-config" --link-static --libs support core)
SYS_LIBS="-lpthread -ldl -lm -lz"
# Link against the umbrella libMLIR (dylib). conda-forge ships a versioned
# soname (e.g. libMLIR.19.1.dylib / libMLIR.so.19.1) without a stable symlink,
# so locate it dynamically. Using the umbrella shared lib means we pick up
# every dialect (Func, Arith, MemRef, SCF, CF, ...) that the upstream backend
# registers.
case "$(uname)" in
    Darwin) MLIR_DYLIB=$(ls "$CONDA_PREFIX"/lib/libMLIR.*.dylib 2>/dev/null | head -1) ;;
    *)      MLIR_DYLIB=$(ls "$CONDA_PREFIX"/lib/libMLIR.so.* 2>/dev/null | head -1) ;;
esac
if [ -z "$MLIR_DYLIB" ]; then
    echo "error: could not find libMLIR shared library in $CONDA_PREFIX/lib" >&2
    exit 1
fi
MLIR_LIBS="$MLIR_DYLIB"

COREC_C_FILES="corec/base/io.c corec/base/buddy.c corec/base/arena.c corec/base/scratch.c corec/base/format.c corec/base/math.c corec/base/string.c corec/base/mem.c corec/base/numconv.c corec/base/assert.c corec/base/exit.c"
PROJ_C_FILES="tests/cross/driver.c mlir_generic_printer.c mlir_op_names.c"

# corec base sources can be compiled normally; only platform_*.c needs
# PLATFORM_SKIP_ENTRY so libc supplies _start / main wrapper instead.
$CC -c -g -I corec -I . $COREC_C_FILES $PROJ_C_FILES tests/upstream_main.c
$CC -c -g -I corec -I . -DPLATFORM_SKIP_ENTRY $PLATFORM_C
$CXX -c -std=c++17 -fno-rtti -g -I corec -I . -I "$CONDA_PREFIX/include" mlir_api_impl_upstream.cpp
$CXX -g -o cross_upstream \
    upstream_main.o driver.o mlir_generic_printer.o mlir_op_names.o mlir_api_impl_upstream.o \
    io.o buddy.o arena.o scratch.o format.o math.o string.o mem.o numconv.o assert.o exit.o $PLATFORM_OBJ \
    -L "$CONDA_PREFIX/lib" $MLIR_LIBS $LLVM_LIBS $SYS_LIBS \
    -Wl,-rpath,"$CONDA_PREFIX/lib"

#!/usr/bin/env bash
set -e

LLVM_LIBS=$("$CONDA_PREFIX/bin/llvm-config" --link-static --libs support core)
SYS_LIBS=$("$CONDA_PREFIX/bin/llvm-config" --system-libs)
MLIR_LIBS="-lMLIRIR -lMLIRSupport"

COREC_C_FILES="corec/base/io.c corec/base/buddy.c corec/base/arena.c corec/base/scratch.c corec/base/format.c corec/base/math.c corec/base/string.c corec/base/mem.c corec/base/numconv.c corec/base/assert.c corec/base/exit.c $PLATFORM_C"
PROJ_C_FILES="tests/cross/driver.c mlir_generic_printer.c mlir_op_names.c"

$CC -c -g -I corec -I . $COREC_C_FILES $PROJ_C_FILES
$CXX -c -std=c++17 -fno-rtti -g -I corec -I . -I "$CONDA_PREFIX/include" mlir_api_impl_upstream.cpp
$CXX -g -o cross_upstream \
    driver.o mlir_generic_printer.o mlir_op_names.o mlir_api_impl_upstream.o \
    io.o buddy.o arena.o scratch.o format.o math.o string.o mem.o numconv.o assert.o exit.o $PLATFORM_OBJ \
    -L "$CONDA_PREFIX/lib" $MLIR_LIBS $LLVM_LIBS $SYS_LIBS $EXTRA_LINK_FLAGS \
    -Wl,-rpath,"$CONDA_PREFIX/lib"

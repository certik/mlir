#!/usr/bin/env bash
set -e

LLVM_LIBS=$("$CONDA_PREFIX/bin/llvm-config" --link-static --libs support core analysis transformutils frontendopenmp webassemblycodegen webassemblyasmparser webassemblydesc webassemblydisassembler webassemblyinfo webassemblyutils target mc mcparser asmprinter codegen selectiondag globalisel bitwriter)
SYS_LIBS="-lpthread -ldl -lm $CONDA_PREFIX/lib/libz.a -lzstd"
# GNU ld requires --start-group/--end-group to resolve cyclic dependencies
# between static archives. Apple's ld64 resolves these without help.
case "$(uname)" in
    Linux) GROUP_START="-Wl,--start-group"; GROUP_END="-Wl,--end-group" ;;
    *)     GROUP_START="";                  GROUP_END="" ;;
esac
# Link MLIR statically: enumerate every libMLIR*.a archive shipped by
# conda-forge. This pulls in all dialects (Func, Arith, MemRef, SCF, CF, ...)
# the upstream backend registers.
MLIR_LIBS=$(ls "$CONDA_PREFIX"/lib/libMLIR*.a 2>/dev/null)
if [ -z "$MLIR_LIBS" ]; then
    echo "error: could not find any libMLIR*.a static archives in $CONDA_PREFIX/lib" >&2
    exit 1
fi

COREC_C_FILES="corec/base/io.c corec/base/buddy.c corec/base/arena.c corec/base/scratch.c corec/base/format.c corec/base/math.c corec/base/string.c corec/base/mem.c corec/base/numconv.c corec/base/assert.c corec/base/exit.c"
PROJ_C_FILES="tests/cross/driver.c mlir_generic_printer.c mlir_op_names.c mlir_lower_to_llvm.c mlir_translate_to_llvm_ir.c mlir_wasm_to_wat.c mlir_llvm_to_wasmssa.c mlir_wasmssa_to_wasmstack.c mlir_wasmstack_to_bin.c"

# corec base sources can be compiled normally; only platform_*.c needs
# PLATFORM_SKIP_ENTRY so libc supplies _start / main wrapper instead.
$CC -c -g -I corec -I . $COREC_C_FILES $PROJ_C_FILES tests/upstream_main.c
$CC -c -g -I corec -I . -DPLATFORM_SKIP_ENTRY $PLATFORM_C
$CXX -c -std=c++17 -fno-rtti -g -I corec -I . -I "$CONDA_PREFIX/include" mlir_api_impl_upstream.cpp
$CXX -g -o cross_upstream \
    upstream_main.o driver.o mlir_generic_printer.o mlir_op_names.o mlir_lower_to_llvm.o mlir_translate_to_llvm_ir.o mlir_wasm_to_wat.o mlir_llvm_to_wasmssa.o mlir_wasmssa_to_wasmstack.o mlir_wasmstack_to_bin.o mlir_api_impl_upstream.o \
    io.o buddy.o arena.o scratch.o format.o math.o string.o mem.o numconv.o assert.o exit.o $PLATFORM_OBJ \
    -L "$CONDA_PREFIX/lib" $GROUP_START $MLIR_LIBS $LLVM_LIBS $GROUP_END $SYS_LIBS \
    -Wl,-rpath,"$CONDA_PREFIX/lib"

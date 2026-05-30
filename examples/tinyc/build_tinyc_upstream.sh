#!/usr/bin/env bash
set -e

# Build the tinyC example compiler against the upstream MLIR backend.
# Mirrors tests/build_parser_upstream.sh — same link line, just
# substituting tinyC sources + driver for the parser.

LLVM_LIBS=$("$CONDA_PREFIX/bin/llvm-config" --link-static --libs support core analysis transformutils frontendopenmp webassemblycodegen webassemblyasmparser webassemblydesc webassemblydisassembler webassemblyinfo webassemblyutils target mc mcparser asmprinter codegen selectiondag globalisel bitwriter)
SYS_LIBS="-lpthread -ldl -lm $CONDA_PREFIX/lib/libz.a -lzstd"
case "$(uname)" in
    Linux) GROUP_START="-Wl,--start-group"; GROUP_END="-Wl,--end-group" ;;
    *)     GROUP_START="";                  GROUP_END="" ;;
esac
MLIR_LIBS=$(ls "$CONDA_PREFIX"/lib/libMLIR*.a 2>/dev/null)
if [ -z "$MLIR_LIBS" ]; then
    echo "error: could not find any libMLIR*.a static archives in $CONDA_PREFIX/lib" >&2
    exit 1
fi

COREC_C_FILES="corec/base/io.c corec/base/buddy.c corec/base/arena.c corec/base/scratch.c corec/base/format.c corec/base/math.c corec/base/string.c corec/base/strbuf.c corec/base/mem.c corec/base/numconv.c corec/base/assert.c corec/base/exit.c"
TINYC_C_FILES="examples/tinyc/lex.c examples/tinyc/preprocess.c examples/tinyc/parse.c examples/tinyc/emit.c examples/tinyc/driver.c mlir_op_names.c mlir_lower_to_llvm.c mlir_translate_to_llvm_ir.c mlir_translate_to_wasm.c mlir_wasm_to_wat.c mlir_wasm_to_macho.c mlir_llvm_to_wasmssa.c mlir_wasmssa_to_wasmstack.c mlir_wasmstack_to_bin.c mlir_wasm_link.c mlir_wasm_to_wasmstack.c mlir_wasmstack_to_wasmssa.c mlir_wasmssa_to_wmir.c mlir_wmir_mem2reg.c mlir_wmir_regalloc.c mlir_wmir_to_aarch64.c mlir_aarch64_to_macho.c mlir_generic_printer.c mlir_lift_cf_to_scf.c"

$CC -c -g -DTINYC_HAS_UPSTREAM -I corec -I . $COREC_C_FILES $TINYC_C_FILES tests/upstream_main.c
$CC -c -g -I corec -I . -DPLATFORM_SKIP_ENTRY $PLATFORM_C
$CXX -c -std=c++17 -fno-rtti -g -I corec -I . -I "$CONDA_PREFIX/include" mlir_api_impl_upstream.cpp
$CXX -g -o tinyc \
    upstream_main.o lex.o preprocess.o parse.o emit.o driver.o mlir_api_impl_upstream.o mlir_op_names.o mlir_lower_to_llvm.o mlir_translate_to_llvm_ir.o mlir_translate_to_wasm.o mlir_wasm_to_wat.o mlir_wasm_to_macho.o mlir_llvm_to_wasmssa.o mlir_wasmssa_to_wasmstack.o mlir_wasmstack_to_bin.o mlir_wasm_link.o mlir_wasm_to_wasmstack.o mlir_wasmstack_to_wasmssa.o mlir_wasmssa_to_wmir.o mlir_wmir_mem2reg.o mlir_wmir_regalloc.o mlir_wmir_to_aarch64.o mlir_aarch64_to_macho.o mlir_generic_printer.o mlir_lift_cf_to_scf.o \
    io.o buddy.o arena.o scratch.o format.o math.o string.o strbuf.o mem.o numconv.o assert.o exit.o $PLATFORM_OBJ \
    -L "$CONDA_PREFIX/lib" $GROUP_START $MLIR_LIBS $LLVM_LIBS $GROUP_END $SYS_LIBS \
    -Wl,-rpath,"$CONDA_PREFIX/lib"

#!/usr/bin/env bash
set -e

# Optimized build of the tinyC example compiler against the NATIVE MLIR
# backend (mlir_api_impl.c). Mirrors build_tinyc_native.sh but compiles
# with -O3 -flto -DNDEBUG to produce the fastest possible binary. Used
# by the bench_tinyc_compile benchmark to measure raw native-pipeline
# throughput.
#
# Output binary: ./tinyc_native_opt.

case "$(uname)" in
    Linux) GROUP_START="-Wl,--start-group"; GROUP_END="-Wl,--end-group" ;;
    *)     GROUP_START="";                  GROUP_END="" ;;
esac

OPT_FLAGS="-O3 -DNDEBUG -flto"

COREC_C_FILES="corec/base/io.c corec/base/buddy.c corec/base/arena.c corec/base/scratch.c corec/base/format.c corec/base/math.c corec/base/string.c corec/base/strbuf.c corec/base/mem.c corec/base/numconv.c corec/base/assert.c corec/base/exit.c"
TINYC_C_FILES="examples/tinyc/lex.c examples/tinyc/preprocess.c examples/tinyc/parse.c examples/tinyc/emit.c examples/tinyc/driver.c"
NATIVE_C_FILES="mlir_api_impl.c mlir_op_names.c mlir_lower_to_llvm.c mlir_translate_to_llvm_ir.c mlir_translate_to_wasm.c mlir_wasm_to_wat.c mlir_wasm_to_macho.c mlir_llvm_to_wasmssa.c mlir_wasmssa_to_wasmstack.c mlir_wasmstack_to_bin.c mlir_wasm_link.c mlir_wasm_to_wasmstack.c mlir_wasmstack_to_wasmssa.c mlir_wasmssa_to_wmir.c mlir_llvm_mem2reg.c mlir_wmir_mem2reg.c mlir_wmir_regalloc.c mlir_wmir_to_aarch64.c mlir_llvm_to_aarch64.c mlir_aarch64_to_macho.c tokenizer.c mlir_parser.c op_parsers.c mlir_classic_printer.c mlir_generic_printer.c mlir_lift_cf_to_scf.c"

OBJ_DIR="build_opt_native"
rm -rf "$OBJ_DIR"
mkdir -p "$OBJ_DIR"

$CC $OPT_FLAGS -c -I corec -I . -o "$OBJ_DIR/upstream_main.o" tests/upstream_main.c
for f in $COREC_C_FILES $TINYC_C_FILES $NATIVE_C_FILES; do
    base="$(basename "$f" .c).o"
    $CC $OPT_FLAGS -c -I corec -I . -o "$OBJ_DIR/$base" "$f"
done
$CC $OPT_FLAGS -c -I corec -I . -DPLATFORM_SKIP_ENTRY -o "$OBJ_DIR/$PLATFORM_OBJ" $PLATFORM_C

$CC $OPT_FLAGS -o tinyc_native_opt \
    "$OBJ_DIR"/upstream_main.o \
    "$OBJ_DIR"/lex.o "$OBJ_DIR"/preprocess.o "$OBJ_DIR"/parse.o "$OBJ_DIR"/emit.o "$OBJ_DIR"/driver.o \
    "$OBJ_DIR"/mlir_api_impl.o "$OBJ_DIR"/mlir_op_names.o "$OBJ_DIR"/mlir_lower_to_llvm.o \
    "$OBJ_DIR"/mlir_translate_to_llvm_ir.o "$OBJ_DIR"/mlir_translate_to_wasm.o "$OBJ_DIR"/mlir_wasm_to_wat.o \
    "$OBJ_DIR"/mlir_wasm_to_macho.o \
    "$OBJ_DIR"/mlir_llvm_to_wasmssa.o "$OBJ_DIR"/mlir_wasmssa_to_wasmstack.o "$OBJ_DIR"/mlir_wasmstack_to_bin.o \
    "$OBJ_DIR"/mlir_wasm_link.o "$OBJ_DIR"/mlir_wasm_to_wasmstack.o "$OBJ_DIR"/mlir_wasmstack_to_wasmssa.o "$OBJ_DIR"/mlir_wasmssa_to_wmir.o "$OBJ_DIR"/mlir_llvm_mem2reg.o mlir_wmir_mem2reg.o "$OBJ_DIR"/mlir_wmir_regalloc.o "$OBJ_DIR"/mlir_wmir_to_aarch64.o mlir_llvm_to_aarch64.o \
    "$OBJ_DIR"/mlir_aarch64_to_macho.o \
    "$OBJ_DIR"/tokenizer.o "$OBJ_DIR"/mlir_parser.o "$OBJ_DIR"/op_parsers.o \
    "$OBJ_DIR"/mlir_classic_printer.o "$OBJ_DIR"/mlir_generic_printer.o "$OBJ_DIR"/mlir_lift_cf_to_scf.o \
    "$OBJ_DIR"/io.o "$OBJ_DIR"/buddy.o "$OBJ_DIR"/arena.o "$OBJ_DIR"/scratch.o "$OBJ_DIR"/format.o \
    "$OBJ_DIR"/math.o "$OBJ_DIR"/string.o "$OBJ_DIR"/strbuf.o "$OBJ_DIR"/mem.o "$OBJ_DIR"/numconv.o \
    "$OBJ_DIR"/assert.o "$OBJ_DIR"/exit.o "$OBJ_DIR"/$PLATFORM_OBJ \
    $GROUP_START $GROUP_END -lm

echo "Built tinyc_native_opt with -O3 -flto."
ls -l tinyc_native_opt

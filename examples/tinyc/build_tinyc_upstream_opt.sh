#!/usr/bin/env bash
set -e

# Optimized build of the tinyC example compiler against the upstream
# MLIR backend (mlir_api_impl_upstream.cpp). Mirrors
# build_tinyc_upstream.sh but compiles every translation unit with
# -O3 -DNDEBUG to produce the fastest possible binary. Used by the
# bench_tinyc_compile benchmark.
#
# The output binary supports both --lowering=native (our own MLIR ->
# LLVM IR pipeline) and --lowering=upstream (upstream's pass pipeline
# + LLVM translator), so the same binary is benchmarked under both
# configurations.
#
# Output binary: ./tinyc_upstream_opt.

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

# Note: do NOT use -flto here. With the LLVM/MLIR static archives, LTO
# blows up link memory above 16 GB on macOS and is not the variable we
# want to measure (these libs are themselves built without LTO). Per-TU
# -O3 -DNDEBUG already removes all asserts and unlocks aggressive opts.
OPT_FLAGS="-O3 -DNDEBUG"

COREC_C_FILES="corec/base/io.c corec/base/buddy.c corec/base/arena.c corec/base/scratch.c corec/base/format.c corec/base/math.c corec/base/string.c corec/base/strbuf.c corec/base/mem.c corec/base/numconv.c corec/base/assert.c corec/base/exit.c"
TINYC_C_FILES="examples/tinyc/lex.c examples/tinyc/preprocess.c examples/tinyc/parse.c examples/tinyc/emit.c examples/tinyc/driver.c mlir_op_names.c mlir_lower_to_llvm.c mlir_translate_to_llvm_ir.c mlir_translate_to_wasm.c mlir_wasm_to_wat.c mlir_wasm_to_macho.c mlir_llvm_to_wasmssa.c mlir_wasmssa_to_wasmstack.c mlir_wasmstack_to_bin.c mlir_wasm_link.c mlir_wasmssa_to_wmir.c mlir_wmir_to_aarch64.c mlir_aarch64_to_macho.c mlir_generic_printer.c mlir_lift_cf_to_scf.c"

OBJ_DIR="build_opt_upstream"
rm -rf "$OBJ_DIR"
mkdir -p "$OBJ_DIR"

$CC $OPT_FLAGS -c -DTINYC_HAS_UPSTREAM -I corec -I . -o "$OBJ_DIR/upstream_main.o" tests/upstream_main.c
for f in $COREC_C_FILES $TINYC_C_FILES; do
    base="$(basename "$f" .c).o"
    $CC $OPT_FLAGS -c -DTINYC_HAS_UPSTREAM -I corec -I . -o "$OBJ_DIR/$base" "$f"
done
$CC $OPT_FLAGS -c -I corec -I . -DPLATFORM_SKIP_ENTRY -o "$OBJ_DIR/$PLATFORM_OBJ" $PLATFORM_C
$CXX $OPT_FLAGS -c -std=c++17 -fno-rtti -I corec -I . -I "$CONDA_PREFIX/include" \
    -o "$OBJ_DIR/mlir_api_impl_upstream.o" mlir_api_impl_upstream.cpp

$CXX $OPT_FLAGS -o tinyc_upstream_opt \
    "$OBJ_DIR"/upstream_main.o \
    "$OBJ_DIR"/lex.o "$OBJ_DIR"/preprocess.o "$OBJ_DIR"/parse.o "$OBJ_DIR"/emit.o "$OBJ_DIR"/driver.o \
    "$OBJ_DIR"/mlir_api_impl_upstream.o "$OBJ_DIR"/mlir_op_names.o "$OBJ_DIR"/mlir_lower_to_llvm.o \
    "$OBJ_DIR"/mlir_translate_to_llvm_ir.o "$OBJ_DIR"/mlir_translate_to_wasm.o "$OBJ_DIR"/mlir_wasm_to_wat.o \
    "$OBJ_DIR"/mlir_wasm_to_macho.o \
    "$OBJ_DIR"/mlir_llvm_to_wasmssa.o "$OBJ_DIR"/mlir_wasmssa_to_wasmstack.o "$OBJ_DIR"/mlir_wasmstack_to_bin.o \
    "$OBJ_DIR"/mlir_wasm_link.o "$OBJ_DIR"/mlir_wasmssa_to_wmir.o "$OBJ_DIR"/mlir_wmir_to_aarch64.o \
    "$OBJ_DIR"/mlir_aarch64_to_macho.o \
    "$OBJ_DIR"/mlir_generic_printer.o "$OBJ_DIR"/mlir_lift_cf_to_scf.o \
    "$OBJ_DIR"/io.o "$OBJ_DIR"/buddy.o "$OBJ_DIR"/arena.o "$OBJ_DIR"/scratch.o "$OBJ_DIR"/format.o \
    "$OBJ_DIR"/math.o "$OBJ_DIR"/string.o "$OBJ_DIR"/strbuf.o "$OBJ_DIR"/mem.o "$OBJ_DIR"/numconv.o \
    "$OBJ_DIR"/assert.o "$OBJ_DIR"/exit.o "$OBJ_DIR"/$PLATFORM_OBJ \
    -L "$CONDA_PREFIX/lib" $GROUP_START $MLIR_LIBS $LLVM_LIBS $GROUP_END $SYS_LIBS \
    -Wl,-rpath,"$CONDA_PREFIX/lib"

echo "Built tinyc_upstream_opt with -O3."
ls -l tinyc_upstream_opt

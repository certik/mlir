#!/usr/bin/env bash
set -euo pipefail

# Build and run the #357 acceptance tests (ports of LFortran's llvm_rsqrt
# and llvm_reductions passes) against the native backend, and (when run in
# the `upstream` pixi environment with libMLIR*.a present) also against the
# upstream MLIR 19 backend. Pass `--bench` to also run timing benchmarks.

BENCH=0
if [ "${1:-}" = "--bench" ]; then
    BENCH=1
fi

CC=${CC:-clang}
CXX=${CXX:-clang++}
COREC="corec/base/io.c corec/base/buddy.c corec/base/arena.c corec/base/scratch.c corec/base/format.c corec/base/math.c corec/base/string.c corec/base/strbuf.c corec/base/mem.c corec/base/numconv.c corec/base/assert.c corec/base/exit.c"
ACC="tests/acceptance/test_acceptance.c tests/acceptance/pass_llvm_rsqrt.c tests/acceptance/pass_llvm_reductions.c"
SHARED="mlir_op_names.c mlir_generic_printer.c mlir_lift_cf_to_scf.c"

case "$(uname)" in
    Darwin) PLATFORM_C="corec/platform/platform_macos.c"; GROUP_START=""; GROUP_END="" ;;
    *)      PLATFORM_C="corec/platform/platform_linux.c"; GROUP_START="-Wl,--start-group"; GROUP_END="-Wl,--end-group" ;;
esac

TMPDIR_ACC=$(mktemp -d)
trap 'rm -rf "$TMPDIR_ACC"' EXIT

$CC -O2 -DMLIR_API_CHECKS -I corec -I . -I tests/acceptance -DPLATFORM_SKIP_ENTRY \
    -o "$TMPDIR_ACC/acc_native" $ACC mlir_api_impl.c $SHARED $COREC "$PLATFORM_C" -lm
"$TMPDIR_ACC/acc_native" > "$TMPDIR_ACC/out_native.txt"
diff -u tests/acceptance/expected.txt "$TMPDIR_ACC/out_native.txt"
echo "acceptance (native): PASS"
if [ "$BENCH" = "1" ]; then
    "$TMPDIR_ACC/acc_native" --bench
fi

if [ -n "${CONDA_PREFIX:-}" ] && ls "$CONDA_PREFIX"/lib/libMLIR*.a >/dev/null 2>&1; then
    LLVM_LIBS=$("$CONDA_PREFIX/bin/llvm-config" --link-static --libs support core analysis transformutils frontendopenmp webassemblycodegen webassemblyasmparser webassemblydesc webassemblydisassembler webassemblyinfo webassemblyutils target mc mcparser asmprinter codegen selectiondag globalisel bitwriter)
    MLIR_LIBS=$(ls "$CONDA_PREFIX"/lib/libMLIR*.a)
    SYS_LIBS="-lpthread -ldl -lm $CONDA_PREFIX/lib/libz.a -lzstd"
    mkdir -p "$TMPDIR_ACC/up_objs"
    for f in $ACC $SHARED $COREC "$PLATFORM_C"; do
        $CC -O2 -I corec -I . -I tests/acceptance -DPLATFORM_SKIP_ENTRY \
            -c "$f" -o "$TMPDIR_ACC/up_objs/$(basename "$f" .c).o"
    done
    $CXX -O2 -std=c++17 -fno-rtti -DMLIR_API_CHECKS -I corec -I . -I "$CONDA_PREFIX/include" \
        -c mlir_api_impl_upstream.cpp -o "$TMPDIR_ACC/up_objs/up.o"
    $CXX -O2 -o "$TMPDIR_ACC/acc_upstream" "$TMPDIR_ACC"/up_objs/*.o \
        -L "$CONDA_PREFIX/lib" $GROUP_START $MLIR_LIBS $LLVM_LIBS $GROUP_END $SYS_LIBS \
        -Wl,-rpath,"$CONDA_PREFIX/lib"
    "$TMPDIR_ACC/acc_upstream" > "$TMPDIR_ACC/out_upstream.txt"
    diff -u tests/acceptance/expected.txt "$TMPDIR_ACC/out_upstream.txt"
    echo "acceptance (upstream): PASS"
    if [ "$BENCH" = "1" ]; then
        "$TMPDIR_ACC/acc_upstream" --bench
    fi
fi

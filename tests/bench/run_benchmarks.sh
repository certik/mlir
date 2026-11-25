#!/usr/bin/env bash
set -euo pipefail

# Build and run the #343 / #358 microbenchmarks:
#   1. bench_native   (public C API on mlir_api_impl.c)
#   2. bench_upstream (public C API on mlir_api_impl_upstream.cpp)
#   3. bench_mlir_cpp (direct MLIR 19 C++ API baseline)
# All compiled at -O2 without MLIR_API_CHECKS.

CC=${CC:-clang}
CXX=${CXX:-clang++}
COREC="corec/base/io.c corec/base/buddy.c corec/base/arena.c corec/base/scratch.c corec/base/format.c corec/base/math.c corec/base/string.c corec/base/strbuf.c corec/base/mem.c corec/base/numconv.c corec/base/assert.c corec/base/exit.c"
SHARED="mlir_op_names.c mlir_generic_printer.c mlir_lift_cf_to_scf.c"

case "$(uname)" in
    Darwin) PLATFORM_C="corec/platform/platform_macos.c"; GROUP_START=""; GROUP_END="" ;;
    *)      PLATFORM_C="corec/platform/platform_linux.c"; GROUP_START="-Wl,--start-group"; GROUP_END="-Wl,--end-group" ;;
esac

TMPDIR_B=$(mktemp -d)
trap 'rm -rf "$TMPDIR_B"' EXIT

$CC -O2 -I corec -I . -DPLATFORM_SKIP_ENTRY \
    -o "$TMPDIR_B/bench_native" tests/bench/bench_api.c mlir_api_impl.c $SHARED $COREC "$PLATFORM_C" -lm
"$TMPDIR_B/bench_native" native

# Unity-build (inlined) native: bench_api.c uses ONLY mlir_api.h, while the
# implementation is included underneath so the compiler can inline API calls.
cat > "$TMPDIR_B/unity_native.c" <<'EOF'
#include "mlir_api_impl.c"
#include "mlir_op_names.c"
#include "tests/bench/bench_api.c"
EOF
$CC -O2 -I corec -I . -DPLATFORM_SKIP_ENTRY \
    -o "$TMPDIR_B/bench_native_inl" "$TMPDIR_B/unity_native.c" mlir_generic_printer.c mlir_lift_cf_to_scf.c $COREC "$PLATFORM_C" -lm
"$TMPDIR_B/bench_native_inl" native_inl

if [ -n "${CONDA_PREFIX:-}" ] && ls "$CONDA_PREFIX"/lib/libMLIR*.a >/dev/null 2>&1; then
    LLVM_LIBS=$("$CONDA_PREFIX/bin/llvm-config" --link-static --libs support core analysis transformutils frontendopenmp webassemblycodegen webassemblyasmparser webassemblydesc webassemblydisassembler webassemblyinfo webassemblyutils target mc mcparser asmprinter codegen selectiondag globalisel bitwriter)
    MLIR_LIBS=$(ls "$CONDA_PREFIX"/lib/libMLIR*.a)
    SYS_LIBS="-lpthread -ldl -lm $CONDA_PREFIX/lib/libz.a -lzstd"
    mkdir -p "$TMPDIR_B/up_objs"
    for f in tests/bench/bench_api.c $SHARED $COREC "$PLATFORM_C"; do
        $CC -O2 -I corec -I . -DPLATFORM_SKIP_ENTRY \
            -c "$f" -o "$TMPDIR_B/up_objs/$(basename "$f" .c).o"
    done
    $CXX -O2 -std=c++17 -fno-rtti -I corec -I . -I "$CONDA_PREFIX/include" \
        -c mlir_api_impl_upstream.cpp -o "$TMPDIR_B/up_objs/up.o"
    $CXX -O2 -o "$TMPDIR_B/bench_upstream" "$TMPDIR_B"/up_objs/*.o \
        -L "$CONDA_PREFIX/lib" $GROUP_START $MLIR_LIBS $LLVM_LIBS $GROUP_END $SYS_LIBS \
        -Wl,-rpath,"$CONDA_PREFIX/lib"
    "$TMPDIR_B/bench_upstream" upstream

    # Unity-build (inlined) upstream: bench_api.c uses ONLY mlir_api.h, while
    # mlir_api_impl_upstream.cpp is included underneath for full inlining.
    cat > "$TMPDIR_B/unity_upstream.cpp" <<'EOF'
#include "mlir_api_impl_upstream.cpp"
#include "tests/bench/bench_api.c"
EOF
    $CXX -O2 -std=c++17 -fno-rtti -I corec -I . -I "$CONDA_PREFIX/include" -DPLATFORM_SKIP_ENTRY \
        -c "$TMPDIR_B/unity_upstream.cpp" -o "$TMPDIR_B/up_inl.o"
    rm -f "$TMPDIR_B/up_objs/bench_api.o" "$TMPDIR_B/up_objs/up.o"
    $CXX -O2 -o "$TMPDIR_B/bench_upstream_inl" "$TMPDIR_B/up_inl.o" "$TMPDIR_B"/up_objs/*.o \
        -L "$CONDA_PREFIX/lib" $GROUP_START $MLIR_LIBS $LLVM_LIBS $GROUP_END $SYS_LIBS \
        -Wl,-rpath,"$CONDA_PREFIX/lib"
    "$TMPDIR_B/bench_upstream_inl" up_inl

    $CXX -O2 -std=c++17 -fno-rtti -I "$CONDA_PREFIX/include" \
        -o "$TMPDIR_B/bench_mlir_cpp" tests/bench/bench_mlir_cpp.cpp \
        -L "$CONDA_PREFIX/lib" $GROUP_START $MLIR_LIBS $LLVM_LIBS $GROUP_END $SYS_LIBS \
        -Wl,-rpath,"$CONDA_PREFIX/lib"
    "$TMPDIR_B/bench_mlir_cpp"
fi

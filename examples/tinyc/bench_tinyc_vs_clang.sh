#!/usr/bin/env bash
#
# bench_tinyc_vs_clang.sh — two head-to-head benchmarks (Darwin/arm64 only):
#
#   BENCHMARK 1 — compiler speed (tinyc vs clang, same compile job)
#     A) build tinyc with clang -O3 -flto (= tinyc_native_opt, the fastest
#        tinyc binary) and time how long that fast tinyc takes to compile
#        the whole tinyc source tree into a native Mach-O binary
#        (the macho-via-llvm self-host job).
#     B) time how long clang -O0 takes to compile the tinyc source tree into
#        a native tinyc binary (build_tinyc_native.sh).
#     This compares the *compiler* speed of tinyc against clang fairly: both
#     compile the tinyc compiler's own sources.
#
#   BENCHMARK 2 — generated-code quality (tinyc macho backend vs clang -O0)
#     C) take the clang-O0-built tinyc (= tinyc_native) and time how long it
#        takes to compile the tinyc source tree (same self-host job as 1A).
#     D) take stage1 — tinyc compiled by tinyc itself through the new Mach-O
#        backend (= tinyc_stage1_macho_llvm) — and time the same job.
#     C and D are the *same* program built two ways: C's machine code comes
#     from clang -O0, D's from tinyc's own llvm->aarch64->Mach-O backend.
#     The ratio measures how good tinyc's generated code is vs clang -O0.
#
# Usage:
#   pixi run -e upstream bench_tinyc_vs_clang [--repeats N] [--rebuild]
#
#   --repeats N   best-of-N wall-clock timing per job (default 3)
#   --rebuild     force-rebuild every prerequisite binary before timing
#
# All timed self-host runs use *native* Mach-O tinyc binaries (no wasmtime in
# the timed path), so the numbers reflect native execution speed only.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [ "$(uname)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "error: this benchmark is Darwin/arm64 only (uses the Mach-O backend)." >&2
    exit 1
fi

REPEATS=3
REBUILD=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --repeats) REPEATS="$2"; shift 2 ;;
        --repeats=*) REPEATS="${1#*=}"; shift ;;
        --rebuild) REBUILD=1; shift ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# --- timing helper -----------------------------------------------------------
TIMER="$(mktemp -t tinyc_bench_timer.XXXX.py)"
cat > "$TIMER" <<'PY'
import subprocess, sys, time
t0 = time.perf_counter()
r = subprocess.run(sys.argv[1:], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"{time.perf_counter()-t0:.6f}")
sys.exit(r.returncode)
PY
cleanup() { rm -f "$TIMER"; rm -rf "$BENCH_TMP"; }
BENCH_TMP="$(mktemp -d -t tinyc_bench.XXXX)"
trap cleanup EXIT

# best_time N -- cmd args...  -> prints min seconds (6 decimals) on stdout
best_time() {
    local n="$1"; shift
    local times=() t rc i
    for ((i = 1; i <= n; i++)); do
        if ! t="$(python3 "$TIMER" "$@")"; then
            rc=$?
            echo "" >&2
            echo "ERROR: timed command failed (rc=$rc):" >&2
            echo "  $*" >&2
            echo "Re-running once with output for diagnosis:" >&2
            "$@" >&2 || true
            return 1
        fi
        times+=("$t")
    done
    printf '%s\n' "${times[@]}" \
        | python3 -c 'import sys; xs=[float(x) for x in sys.stdin if x.strip()]; print(f"{min(xs):.6f}")'
}

fmt() { printf '%.3f' "$1"; }
ratio() { python3 -c "import sys; a=float('$1'); b=float('$2'); print(f'{a/b:.2f}')"; }

# --- prerequisites -----------------------------------------------------------
need_build() { [ "$REBUILD" = 1 ] || [ ! -e "$1" ]; }

echo "==> Preparing prerequisite binaries (rebuild=$REBUILD)"

# tinyc_native_opt : tinyc built by clang -O3 -flto  (fast tinyc)
if need_build tinyc_native_opt; then
    echo "    building tinyc_native_opt (clang -O3 -flto) ..."
    bash examples/tinyc/build_tinyc_native_opt.sh >/dev/null
fi

# tinyc.wasm + vararg shim + tinyc_stage1_macho_llvm : tinyc-codegen'd tinyc.
# selfhost_tinyc_macho_llvm_stage1 (wasm env) drives tinyc.wasm to recompile
# the tree and lift+lower it to a native Mach-O binary via tinyc's own backend.
if need_build tinyc_stage1_macho_llvm || need_build tinyc.wasm || need_build tinyc_wasm_vararg.wasm.o; then
    echo "    building tinyc.wasm + tinyc_stage1_macho_llvm (tinyc self-codegen) ..."
    pixi run -e wasm selfhost_tinyc_macho_llvm_stage1 >/dev/null
fi

# tinyc_native : tinyc built by clang -O0 (-g, no optimization).
# Built LAST so the binary on disk is exactly the clang-O0 codegen used by 2C.
echo "    building tinyc_native (clang -O0) ..."
bash examples/tinyc/build_tinyc_native.sh >/dev/null

for f in tinyc_native_opt tinyc_native tinyc_stage1_macho_llvm tinyc_wasm_vararg.wasm.o; do
    [ -e "$f" ] || { echo "error: prerequisite '$f' missing after build" >&2; exit 1; }
done

# --- jobs --------------------------------------------------------------------
# The "compile tinyc" workload for a native tinyc binary: recompile the whole
# tinyc source set to per-file .wasm.o, link, and lift+lower to a Mach-O binary
# (examples/tinyc/selfhost_tinyc_macho_llvm.sh). Output + stage dirs are
# throwaway under $BENCH_TMP.
selfhost_job() {  # selfhost_job <tinyc_binary> <tag>
    local bin="$1" tag="$2"
    best_time "$REPEATS" \
        bash examples/tinyc/selfhost_tinyc_macho_llvm.sh \
            "$bin" "$BENCH_TMP/out_$tag" "$BENCH_TMP/stage_$tag"
}

echo
echo "==> Timing (best-of-$REPEATS) ..."

echo "    [1A] tinyc (clang -O3) compiling tinyc ..."
T_OPT_SELFHOST="$(selfhost_job ./tinyc_native_opt opt)"

echo "    [1B] clang -O0 compiling tinyc ..."
T_CLANG_BUILD="$(best_time "$REPEATS" bash examples/tinyc/build_tinyc_native.sh)"

echo "    [2C] tinyc (built by clang -O0) compiling tinyc ..."
T_NATIVE_SELFHOST="$(selfhost_job ./tinyc_native native)"

echo "    [2D] tinyc (built by tinyc's Mach-O backend) compiling tinyc ..."
T_STAGE1_SELFHOST="$(selfhost_job ./tinyc_stage1_macho_llvm stage1)"

# --- report ------------------------------------------------------------------
echo
echo "================================================================"
echo " tinyc vs clang — best-of-$REPEATS wall-clock (Darwin/arm64)"
echo "================================================================"
echo
echo " BENCHMARK 1 — compiler speed (same compile job: the tinyc source tree)"
echo "   tinyc  (clang -O3 -flto) compiling tinyc : $(fmt "$T_OPT_SELFHOST")s   [self-host -> Mach-O]"
echo "   clang  -O0               compiling tinyc : $(fmt "$T_CLANG_BUILD")s   [clang -c + link]"
echo "   ratio tinyc / clang-O0                   : $(ratio "$T_OPT_SELFHOST" "$T_CLANG_BUILD")x   (<1 means tinyc is faster)"
echo
echo " BENCHMARK 2 — generated-code quality (same program, different codegen)"
echo "   tinyc built by clang -O0          self-compile : $(fmt "$T_NATIVE_SELFHOST")s"
echo "   tinyc built by tinyc Mach-O backend self-compile : $(fmt "$T_STAGE1_SELFHOST")s"
echo "   ratio  tinyc-codegen / clang-O0-codegen        : $(ratio "$T_STAGE1_SELFHOST" "$T_NATIVE_SELFHOST")x   (<1 means tinyc's codegen is faster)"
echo "================================================================"

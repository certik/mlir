#!/usr/bin/env bash
#
# bench_tinyc_vs_clang_linux.sh — Linux/x86_64 head-to-head benchmarks:
#
#   BENCHMARK 1 — compiler speed (tinyc vs clang, same compile job)
#     A) build tinyc with clang -O3 -flto (= tinyc_native_opt) and time how
#        long that fast tinyc takes to compile the whole tinyc source tree into
#        a native ELF binary (the selfhost_tinyc_elf job).
#     B) time how long clang -O0 takes to compile the tinyc source tree into a
#        native tinyc binary (build_tinyc_native.sh).
#
#   BENCHMARK 2 — generated-code quality (tinyc ELF backend vs clang -O0)
#     C) take the clang-O0-built tinyc (= tinyc_native) and time how long it
#        takes to compile the tinyc source tree (same self-host job as 1A).
#     D) take stage1 — tinyc compiled by tinyc itself through the new Linux ELF
#        backend (= tinyc_stage1_elf) — and time the same job.
#
# Usage:
#   bash examples/tinyc/bench_tinyc_vs_clang_linux.sh [--repeats N] [--rebuild]

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [ "$(uname)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
    echo "error: this benchmark is Linux/x86_64 only (uses the ELF backend)." >&2
    exit 1
fi

REPEATS=3
REBUILD=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --repeats) REPEATS="$2"; shift 2 ;;
        --repeats=*) REPEATS="${1#*=}"; shift ;;
        --rebuild) REBUILD=1; shift ;;
        -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

TIMER="$(mktemp -t tinyc_bench_timer.XXXX.py)"
cat > "$TIMER" <<'PY'
import subprocess, sys, time
t0 = time.perf_counter()
# Keep stderr visible (progress lines) but discard stdout so timing stays clean.
r = subprocess.run(sys.argv[1:], stdout=subprocess.DEVNULL)
print(f"{time.perf_counter()-t0:.6f}")
sys.exit(r.returncode)
PY
cleanup() { rm -f "$TIMER"; rm -rf "$BENCH_TMP"; }
BENCH_TMP="$(mktemp -d -t tinyc_bench.XXXX)"
trap cleanup EXIT

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
ratio() { python3 -c "a=float('$1'); b=float('$2'); print(f'{a/b:.2f}')"; }
need_build() { [ "$REBUILD" = 1 ] || [ ! -e "$1" ]; }

echo "==> Preparing prerequisite binaries (rebuild=$REBUILD)"

if need_build tinyc_native_opt; then
    echo "    building tinyc_native_opt (clang -O3 -flto) ..."
    bash examples/tinyc/build_tinyc_native_opt.sh >/dev/null
fi

if need_build tinyc_stage1_elf || need_build tinyc_native_opt \
        || { [ -e tinyc_native ] && [ tinyc_native -nt tinyc_native_opt ]; }; then
    echo "    building tinyc_stage1_elf (tinyc self-codegen via ELF) ..."
    # Same as the Darwin Mach-O bench: stage-1 self-host is driven by the
    # release tinyc_native_opt binary. Rebuild it whenever tinyc_native is
    # newer so x64/aarch backend fixes in the tree are not masked by a stale
    # -O3 -flto copy missing recent codegen changes.
    bash examples/tinyc/build_tinyc_native_opt.sh >/dev/null
    bash examples/tinyc/selfhost_tinyc_elf.sh ./tinyc_native_opt tinyc_stage1_elf selfhost_elf_stage1 >/dev/null
fi

echo "    building tinyc_native (clang -O0) ..."
bash examples/tinyc/build_tinyc_native.sh >/dev/null

for f in tinyc_native_opt tinyc_native tinyc_stage1_elf; do
    [ -e "$f" ] || { echo "error: prerequisite '$f' missing after build" >&2; exit 1; }
done

selfhost_job() {
    local bin="$1" tag="$2"
    best_time "$REPEATS" \
        bash examples/tinyc/selfhost_tinyc_elf.sh \
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

echo "    [2D] tinyc (built by tinyc ELF backend) compiling tinyc ..."
T_STAGE1_SELFHOST="$(selfhost_job ./tinyc_stage1_elf stage1)"

echo
echo "================================================================"
echo " tinyc vs clang — best-of-$REPEATS wall-clock (Linux/x86_64)"
echo "================================================================"
echo
echo " BENCHMARK 1 — compiler speed (same compile job: the tinyc source tree)"
echo "   tinyc  (clang -O3 -flto) compiling tinyc : $(fmt "$T_OPT_SELFHOST")s   [self-host -> ELF]"
echo "   clang  -O0               compiling tinyc : $(fmt "$T_CLANG_BUILD")s   [clang -c + link]"
echo "   ratio tinyc / clang-O0                   : $(ratio "$T_OPT_SELFHOST" "$T_CLANG_BUILD")x   (<1 means tinyc is faster)"
echo
echo " BENCHMARK 2 — generated-code quality (same program, different codegen)"
echo "   tinyc built by clang -O0         self-compile : $(fmt "$T_NATIVE_SELFHOST")s"
echo "   tinyc built by tinyc ELF backend self-compile : $(fmt "$T_STAGE1_SELFHOST")s"
echo "   ratio tinyc-codegen / clang-O0-codegen       : $(ratio "$T_STAGE1_SELFHOST" "$T_NATIVE_SELFHOST")x   (<1 means tinyc's codegen is faster)"
echo "================================================================"

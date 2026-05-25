#!/usr/bin/env python3
"""
Benchmark the three tinyc configurations on compiling the tinyc source
itself to wasm32. For each configuration, every .c source file is
compiled individually to a .wasm.o object (timed) and then all objects
are linked into a single tinyc_bench.wasm (also timed). All three
configurations operate on exactly the same source set and produce
exactly the same set of outputs, so the per-file timings and totals are
directly comparable.

Configurations:
  - native           ./tinyc_native_opt --emit=wasm                  (our own backend + our own lowering)
  - upstream_api     ./tinyc_upstream_opt --emit=wasm --lowering=native
                     (upstream API impl, our own lowering to wasm)
  - upstream_full    ./tinyc_upstream_opt --emit=wasm --lowering=upstream
                     (upstream API impl + upstream MLIR -> LLVM IR ->
                     WASM via LLVM's WebAssembly target)

Linking always uses tinyc itself in --link mode (mlir_wasm_link.c).
That keeps every step of the pipeline inside the benchmarked binary,
which is what we want to measure. The link step pulls in
tinyc_wasm_vararg.wasm.o (the tinyc_va_arg_* shim), so the produced
tinyc_bench.wasm is also runnable under wasmtime.

Each (config, file) measurement is repeated --repeats times and the
best run is reported (standard practice for compiler benchmarks: filters
out OS noise without giving outliers undue weight). The per-config
total is the sum of the best per-file times plus the best link time.

Pass `--wasmtime` to run a second stage that uses each produced
tinyc_bench.wasm under wasmtime to recompile every .c file again. The
C source code inside every produced wasm is the same (our native tinyc),
so functionally the three wasm binaries are equivalent — the only thing
that varies is the code-generation quality of the backend that built
the wasm. The wasmtime stage also reports pairwise bit-identity between
the three produced wasm binaries.

Usage:
    python examples/tinyc/bench_tinyc_compile.py            # default 3 repeats
    python examples/tinyc/bench_tinyc_compile.py --repeats 5
    python examples/tinyc/bench_tinyc_compile.py --csv out.csv
    python examples/tinyc/bench_tinyc_compile.py --wasmtime # also run wasmtime stage

Prereqs (pixi takes care of these for you):
    pixi run -e upstream build_tinyc_native_opt
    pixi run -e upstream build_tinyc_upstream_opt
    pixi run build_tinyc_wasm        # needed for the --wasmtime stage
                                     # (provides tinyc_wasm_vararg.wasm.o)
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = ROOT / "bench_tinyc_out"

# Source set mirrors examples/tinyc/build_tinyc_wasm.sh and
# examples/tinyc/selfhost_tinyc_wasm.sh — every .c file that makes up
# the tinyc.wasm self-host build, in the same order. Keep in sync.
COREC_C_FILES = [
    "corec/base/io.c",
    "corec/base/buddy.c",
    "corec/base/arena.c",
    "corec/base/scratch.c",
    "corec/base/format.c",
    "corec/base/math.c",
    "corec/base/string.c",
    "corec/base/strbuf.c",
    "corec/base/mem.c",
    "corec/base/numconv.c",
    "corec/base/assert.c",
    "corec/base/exit.c",
]
COREC_STDLIB_C_FILES = [
    "corec-stdlib/stdlib/stdio.c",
    "corec-stdlib/stdlib/stdlib.c",
    "corec-stdlib/stdlib/printf.c",
    "corec-stdlib/stdlib/string_impl.c",
]
TINYC_C_FILES = [
    "examples/tinyc/lex.c",
    "examples/tinyc/preprocess.c",
    "examples/tinyc/parse.c",
    "examples/tinyc/emit.c",
    "examples/tinyc/driver.c",
]
NATIVE_C_FILES = [
    "mlir_api_impl.c",
    "mlir_op_names.c",
    "mlir_lower_to_llvm.c",
    "mlir_translate_to_llvm_ir.c",
    "mlir_translate_to_wasm.c",
    "mlir_wasm_to_wat.c",
    "mlir_wasm_to_macho.c",
    "mlir_llvm_to_wasmssa.c",
    "mlir_wasmssa_to_wasmstack.c",
    "mlir_wasmstack_to_bin.c",
    "mlir_wasm_link.c",
    "tokenizer.c",
    "mlir_parser.c",
    "op_parsers.c",
    "mlir_classic_printer.c",
    "mlir_generic_printer.c",
    "mlir_lift_cf_to_scf.c",
]
PLATFORM_C_FILES = ["corec/platform/platform_wasm.c"]

ALL_SOURCES = (
    COREC_C_FILES
    + COREC_STDLIB_C_FILES
    + TINYC_C_FILES
    + NATIVE_C_FILES
    + PLATFORM_C_FILES
)

INCLUDE_FLAGS = ["-I", "corec", "-I", "corec-stdlib/stdlib", "-I", "."]


@dataclass
class Config:
    name: str
    binary: Path
    extra_flags: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def cmd_compile(self, src: Path, obj: Path) -> list[str]:
        return [
            str(self.binary),
            "--emit=wasm",
            *self.extra_flags,
            *INCLUDE_FLAGS,
            "-o", str(obj),
            str(src),
        ]

    def cmd_link(self, objs: list[Path], out: Path, extras: list[Path] | None = None) -> list[str]:
        return [
            str(self.binary), "--link",
            "--export=_start",
            "-o", str(out),
            *[str(o) for o in objs],
            *([str(p) for p in (extras or [])]),
        ]

    def env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.extra_env)
        return env


def time_run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[float, subprocess.CompletedProcess]:
    """Run cmd and return (elapsed_seconds, completed_process). Timing
    uses time.perf_counter() around the subprocess.run boundary — the
    Python overhead is identical across configurations so it cancels
    out of any comparison."""
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    t1 = time.perf_counter()
    return t1 - t0, proc


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:8.1f} ms"


def fmt_s(seconds: float) -> str:
    return f"{seconds:7.3f} s"


def bench_compile_one(cfg: Config, src: Path, obj_dir: Path, repeats: int) -> tuple[float, str | None]:
    """Return (best_time, error_message). If the file fails to compile,
    returns (inf, error). The benchmark continues with the next file
    rather than aborting — this lets us report partial coverage when a
    config doesn't support every input."""
    rel = src.relative_to(ROOT)
    flat = str(rel).replace("/", "_").replace("\\", "_")
    obj = obj_dir / f"{flat}.wasm.o"
    env = cfg.env()
    best = float("inf")
    for _ in range(repeats):
        if obj.exists():
            obj.unlink()
        elapsed, proc = time_run(cfg.cmd_compile(src, obj), ROOT, env)
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            tail = err[-1] if err else f"rc={proc.returncode}"
            return float("inf"), tail
        if elapsed < best:
            best = elapsed
    return best, None


def bench_link(cfg: Config, objs: list[Path], out: Path, repeats: int,
               extras: list[Path] | None = None) -> tuple[float, str | None]:
    env = cfg.env()
    best = float("inf")
    for _ in range(repeats):
        if out.exists():
            out.unlink()
        elapsed, proc = time_run(cfg.cmd_link(objs, out, extras), ROOT, env)
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            tail = err[-1] if err else f"rc={proc.returncode}"
            return float("inf"), tail
        if elapsed < best:
            best = elapsed
    return best, None


def discover_configs() -> list[Config]:
    native_opt = ROOT / "tinyc_native_opt"
    upstream_opt = ROOT / "tinyc_upstream_opt"
    missing = [p for p in [native_opt, upstream_opt] if not p.exists()]
    if missing:
        print("error: missing benchmark binaries:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nbuild them with:\n"
            "  pixi run -e upstream build_tinyc_native_opt\n"
            "  pixi run -e upstream build_tinyc_upstream_opt",
            file=sys.stderr,
        )
        sys.exit(1)
    return [
        Config(
            name="native",
            binary=native_opt,
            extra_flags=[],
            description="native API impl + native lowering",
        ),
        Config(
            name="upstream_api",
            binary=upstream_opt,
            extra_flags=["--lowering=native"],
            # Note: a small number of source files exercise constructs
            # that the upstream cf->scf lift produces (e.g. scf.index_switch)
            # which the native wasmssa lowering does not currently accept.
            # The bench script tolerates per-file failures and reports
            # whatever subset does compile.
            extra_env={},
            description="upstream API impl + native lowering",
        ),
        Config(
            name="upstream_full",
            binary=upstream_opt,
            extra_flags=["--lowering=upstream"],
            description="upstream API impl + upstream MLIR/LLVM lowering",
        ),
    ]


# -----------------------------------------------------------------------------
# Wasmtime stage: re-bench using the per-config tinyc.wasm produced above as
# the compiler, executed under wasmtime. Every tinyc.wasm we produce was
# linked with `tinyc_wasm_vararg.wasm.o`, so it runs under wasmtime with no
# extra setup. The C source code inside every tinyc.wasm is the same (the
# native tinyc), so functionally all three are equivalent — the only thing
# that varies is the code quality produced by each backend.
# -----------------------------------------------------------------------------

def _rel(p: Path) -> str:
    """Render p as a path relative to ROOT (since the wasmtime stage
    runs with cwd=ROOT and mounts ROOT as `.`). wasm32-wasi resolves
    paths via preopened FDs + relative components; passing absolute
    paths or paths outside the preopened mount causes the wasm to
    fail to open the file."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        # Fall back to whatever the caller gave us; the wasm will
        # likely fail to open it but we'll get a clearer error.
        return str(p)


def wasmtime_cmd_compile(wasmtime: str, wasm: Path, src: Path, obj: Path) -> list[str]:
    return [
        wasmtime, "--dir", ".",
        _rel(wasm),
        "--emit=wasm",
        "--lowering=native",
        *INCLUDE_FLAGS,
        "-o", _rel(obj),
        _rel(src),
    ]


def wasmtime_cmd_link(wasmtime: str, wasm: Path, objs: list[Path], out: Path,
                      extras: list[Path]) -> list[str]:
    return [
        wasmtime, "--dir", ".",
        _rel(wasm), "--link",
        "--export=_start",
        "-o", _rel(out),
        *[_rel(o) for o in objs],
        *[_rel(p) for p in extras],
    ]


def bench_wasmtime_compile_one(wasmtime: str, wasm: Path, src: Path,
                                obj_dir: Path, repeats: int
                                ) -> tuple[float, str | None]:
    rel = src.relative_to(ROOT)
    flat = str(rel).replace("/", "_").replace("\\", "_")
    obj = obj_dir / f"{flat}.wasm.o"
    best = float("inf")
    for _ in range(repeats):
        if obj.exists():
            obj.unlink()
        elapsed, proc = time_run(wasmtime_cmd_compile(wasmtime, wasm, src, obj), ROOT)
        if proc.returncode != 0:
            tail = _summarize_proc_output(proc)
            return float("inf"), tail
        if elapsed < best:
            best = elapsed
    return best, None


def bench_wasmtime_link(wasmtime: str, wasm: Path, objs: list[Path], out: Path,
                         extras: list[Path], repeats: int
                         ) -> tuple[float, str | None]:
    best = float("inf")
    for _ in range(repeats):
        if out.exists():
            out.unlink()
        elapsed, proc = time_run(wasmtime_cmd_link(wasmtime, wasm, objs, out, extras), ROOT)
        if proc.returncode != 0:
            tail = _summarize_proc_output(proc)
            return float("inf"), tail
        if elapsed < best:
            best = elapsed
    return best, None


def _summarize_proc_output(proc: subprocess.CompletedProcess) -> str:
    """Pick a single representative line from a failed run. tinyc itself
    writes diagnostics to stdout (the wasm has no way to distinguish);
    wasmtime / our linker write to stderr. Prefer stdout if present so
    miscompilation reports come through cleanly; fall back to stderr; fall
    back to rc."""
    out_lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    err_lines = [ln for ln in (proc.stderr or "").splitlines() if ln.strip()]
    if out_lines:
        return out_lines[0]
    if err_lines:
        return err_lines[-1]
    return f"rc={proc.returncode}"


def run_wasmtime_stage(
    configs: list[Config],
    wasmtime: str,
    repeats: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, str]],
           dict[str, float], dict[str, str | None], dict[str, float]]:
    """Run the wasmtime stage. Returns (per-file results, per-file errors,
    link times, link errors, compile totals)."""
    results: dict[str, dict[str, float]] = {c.name: {} for c in configs}
    errors: dict[str, dict[str, str]] = {c.name: {} for c in configs}
    link_times: dict[str, float] = {}
    link_errors: dict[str, str | None] = {}
    compile_totals: dict[str, float] = {}

    vararg = ROOT / "tinyc_wasm_vararg.wasm.o"
    extras = [vararg] if vararg.exists() else []
    if not extras:
        print("warning: tinyc_wasm_vararg.wasm.o not found; the produced "
              "tinyc.wasm may not be runnable under wasmtime. "
              "Run `pixi run build_tinyc_wasm` first.")

    print("\n" + "=" * 90)
    print(" wasmtime stage: use each produced tinyc.wasm under wasmtime to")
    print(" recompile the tinyc source tree to .wasm.o + link.")
    print(" (functionally equivalent inputs — only the code-gen quality of")
    print("  each backend affects the timings below)")
    print("=" * 90)

    for c in configs:
        cfg_dir = BENCH_DIR / c.name
        wasm = cfg_dir / "tinyc_bench.wasm"
        if not wasm.exists():
            print(f"\n[{c.name}] tinyc_bench.wasm not built — skipping wasmtime stage")
            for src in ALL_SOURCES:
                errors[c.name][src] = "no compiler wasm"
                results[c.name][src] = float("inf")
            link_times[c.name] = float("inf")
            link_errors[c.name] = "no compiler wasm"
            compile_totals[c.name] = 0.0
            continue
        out_dir = cfg_dir / "wasmtime_stage"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        print(f"\n== wasmtime + {c.name} ({wasm.relative_to(ROOT)}) ==")
        for src in ALL_SOURCES:
            best, err = bench_wasmtime_compile_one(
                wasmtime, wasm, ROOT / src, out_dir, repeats
            )
            results[c.name][src] = best
            if err is not None:
                errors[c.name][src] = err
                print(f"  {src:55s}  FAIL  {err}")
            else:
                print(f"  {src:55s}  {fmt_ms(best)}")
        compile_totals[c.name] = sum(
            t for t in results[c.name].values() if t != float("inf")
        )
        if errors[c.name]:
            print(
                f"  (link)                                                   SKIPPED "
                f"({len(errors[c.name])} compile failure(s))"
            )
            link_times[c.name] = float("inf")
            link_errors[c.name] = "compile failures"
            print(f"  {'TOTAL compile only':55s}  {fmt_s(compile_totals[c.name])}")
        else:
            objs = [
                out_dir / f"{src.replace('/', '_').replace(os.sep, '_')}.wasm.o"
                for src in ALL_SOURCES
            ]
            out_wasm = out_dir / "tinyc_stage2.wasm"
            link_t, link_err = bench_wasmtime_link(
                wasmtime, wasm, objs, out_wasm, extras, repeats
            )
            link_times[c.name] = link_t
            link_errors[c.name] = link_err
            if link_err is not None:
                print(f"  (link)                                                   FAIL  {link_err}")
            else:
                print(f"  {'(link)':55s}  {fmt_ms(link_t)}")
                print(f"  {'TOTAL':55s}  {fmt_s(compile_totals[c.name] + link_t)}")
    return results, errors, link_times, link_errors, compile_totals


def compare_produced_wasms(configs: list[Config]) -> None:
    """Report sizes of, and pairwise bit-identity between, the
    tinyc_bench.wasm binaries produced by the three configs. The
    interesting pair is native vs upstream_api: they use identical
    lowering (--lowering=native), so functionally equivalent output is
    expected — but the API impls differ slightly in IR construction
    order, so the bytes typically don't match."""
    print("\n" + "=" * 90)
    print(" produced tinyc.wasm comparison")
    print("=" * 90)
    wasms: dict[str, Path] = {}
    for c in configs:
        wasm = BENCH_DIR / c.name / "tinyc_bench.wasm"
        if wasm.exists():
            wasms[c.name] = wasm
            print(f"  {c.name:15s}  {wasm.relative_to(ROOT)}  ({wasm.stat().st_size:>10} bytes)")
        else:
            print(f"  {c.name:15s}  (not built)")
    names = list(wasms.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            a, b = wasms[a_name], wasms[b_name]
            a_bytes = a.read_bytes()
            b_bytes = b.read_bytes()
            if a_bytes == b_bytes:
                verdict = "BIT-IDENTICAL"
            else:
                # Find first differing offset for context.
                n = min(len(a_bytes), len(b_bytes))
                first_diff = next(
                    (k for k in range(n) if a_bytes[k] != b_bytes[k]),
                    n,
                )
                verdict = (
                    f"DIFFER (sizes {len(a_bytes)} vs {len(b_bytes)}, "
                    f"first diff at byte {first_diff})"
                )
            print(f"  {a_name} vs {b_name}: {verdict}")


def print_per_file_table(
    results: dict[str, dict[str, float]],
    errors: dict[str, dict[str, str]],
    configs: list[Config],
) -> None:
    col_width = max(len(c.name) for c in configs) + 2
    name_width = max(len(s) for s in ALL_SOURCES) + 2
    header = "file".ljust(name_width) + "".join(c.name.rjust(col_width + 3) for c in configs)
    print()
    print(header)
    print("-" * len(header))
    for src in ALL_SOURCES:
        row = src.ljust(name_width)
        for c in configs:
            t = results[c.name].get(src)
            if t is None or t == float("inf"):
                row += "    FAIL".rjust(col_width + 3)
            else:
                row += fmt_ms(t).rjust(col_width + 3)
        print(row)
    print("-" * len(header))
    row = "TOTAL compile".ljust(name_width)
    for c in configs:
        total = sum(t for t in results[c.name].values() if t != float("inf"))
        row += fmt_ms(total).rjust(col_width + 3)
    print(row)
    # Failure summary
    any_fail = False
    for c in configs:
        for src, err in errors[c.name].items():
            if not any_fail:
                print("\nfailures:")
                any_fail = True
            print(f"  [{c.name}] {src}: {err}")


def print_link_and_grand_total(
    compile_totals: dict[str, float],
    link_times: dict[str, float],
    link_errors: dict[str, str | None],
    configs: list[Config],
) -> None:
    name_width = 24
    col_width = max(len(c.name) for c in configs) + 2
    print()
    header = "step".ljust(name_width) + "".join(c.name.rjust(col_width + 3) for c in configs)
    print(header)
    print("-" * len(header))
    row = "compile (sum of OK)".ljust(name_width)
    for c in configs:
        row += fmt_s(compile_totals[c.name]).rjust(col_width + 3)
    print(row)
    row = "link".ljust(name_width)
    for c in configs:
        t = link_times.get(c.name, float("inf"))
        if t == float("inf"):
            row += "    FAIL".rjust(col_width + 3)
        else:
            row += fmt_s(t).rjust(col_width + 3)
    print(row)
    print("-" * len(header))
    row = "GRAND TOTAL".ljust(name_width)
    for c in configs:
        t = link_times.get(c.name, 0.0)
        if t == float("inf"):
            t = 0.0
        row += fmt_s(compile_totals[c.name] + t).rjust(col_width + 3)
    print(row)


def print_speedup(
    compile_totals: dict[str, float],
    link_times: dict[str, float],
    configs: list[Config],
) -> None:
    base = "upstream_full"
    if base not in compile_totals:
        return
    print()
    print(f"relative to {base} (lower = faster):")
    base_lt = link_times.get(base, 0.0)
    base_total = compile_totals[base] + (0.0 if base_lt == float("inf") else base_lt)
    for c in configs:
        if c.name == base:
            continue
        lt = link_times.get(c.name, 0.0)
        total = compile_totals[c.name] + (0.0 if lt == float("inf") else lt)
        if total <= 0 or base_total <= 0:
            continue
        ratio = total / base_total
        label = "slower" if ratio >= 1.0 else "faster"
        # If the other config is faster, print speedup as base/other.
        magnitude = ratio if ratio >= 1.0 else (base_total / total)
        print(f"  {c.name:18s} {magnitude:6.2f}x {label}")


def write_csv(
    path: Path,
    results: dict[str, dict[str, float]],
    link_times: dict[str, float],
    configs: list[Config],
) -> None:
    def fmt(t: float) -> str:
        return "FAIL" if t == float("inf") else f"{t:.6f}"

    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file"] + [c.name for c in configs])
        for src in ALL_SOURCES:
            w.writerow([src] + [fmt(results[c.name].get(src, float("inf"))) for c in configs])
        w.writerow(["__link__"] + [fmt(link_times.get(c.name, float("inf"))) for c in configs])
        w.writerow(
            ["__total__"]
            + [
                fmt(
                    sum(t for t in results[c.name].values() if t != float("inf"))
                    + (
                        0.0
                        if link_times.get(c.name, float("inf")) == float("inf")
                        else link_times[c.name]
                    )
                )
                for c in configs
            ]
        )
    print(f"\nCSV written to {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3, help="runs per measurement; best is kept (default 3)")
    ap.add_argument("--csv", type=Path, default=None, help="write detailed timings to this CSV")
    ap.add_argument("--only", action="append", default=None,
                    help="restrict to a specific config name (repeatable)")
    ap.add_argument("--wasmtime", action="store_true",
                    help="after the native bench, also run each produced "
                         "tinyc.wasm under wasmtime to compile the tinyc "
                         "source tree again and report per-file timings. "
                         "Useful for measuring the runtime cost of the "
                         "code produced by each backend.")
    ap.add_argument("--wasmtime-binary", default="wasmtime",
                    help="wasmtime executable to use (default: PATH lookup)")
    args = ap.parse_args()

    configs = discover_configs()
    if args.only:
        configs = [c for c in configs if c.name in set(args.only)]
        if not configs:
            print("error: --only filtered out every configuration", file=sys.stderr)
            return 2

    # Verify all source files exist before we start timing.
    missing_src = [s for s in ALL_SOURCES if not (ROOT / s).exists()]
    if missing_src:
        print("error: missing source files:", file=sys.stderr)
        for s in missing_src:
            print(f"  - {s}", file=sys.stderr)
        return 1

    if BENCH_DIR.exists():
        shutil.rmtree(BENCH_DIR)
    BENCH_DIR.mkdir(parents=True)

    print(f"benchmarking {len(configs)} configuration(s); {args.repeats} repeat(s) per file")
    for c in configs:
        print(f"  - {c.name:15s}  {c.binary}  {' '.join(c.extra_flags) or '(no extra flags)'}")
    print(f"  {len(ALL_SOURCES)} source files\n")

    results: dict[str, dict[str, float]] = {c.name: {} for c in configs}
    errors: dict[str, dict[str, str]] = {c.name: {} for c in configs}
    link_times: dict[str, float] = {}
    link_errors: dict[str, str | None] = {}
    compile_totals: dict[str, float] = {}

    for c in configs:
        cfg_dir = BENCH_DIR / c.name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        print(f"== {c.name} ({c.description}) ==")
        for src in ALL_SOURCES:
            best, err = bench_compile_one(c, ROOT / src, cfg_dir, args.repeats)
            results[c.name][src] = best
            if err is not None:
                errors[c.name][src] = err
                print(f"  {src:55s}  FAIL  {err}")
            else:
                print(f"  {src:55s}  {fmt_ms(best)}")
        # Only link if every per-file compile succeeded.
        compile_totals[c.name] = sum(
            t for t in results[c.name].values() if t != float("inf")
        )
        if errors[c.name]:
            print(
                f"  (link)                                                   SKIPPED "
                f"({len(errors[c.name])} compile failure(s))"
            )
            link_times[c.name] = float("inf")
            link_errors[c.name] = "compile failures"
            print(f"  {'TOTAL compile only':55s}  {fmt_s(compile_totals[c.name])}\n")
        else:
            objs = [
                cfg_dir / f"{src.replace('/', '_').replace(os.sep, '_')}.wasm.o"
                for src in ALL_SOURCES
            ]
            out_wasm = cfg_dir / "tinyc_bench.wasm"
            # Pull in the tinyc_va_arg_* shim so the produced wasm is
            # actually runnable under wasmtime (this is also what
            # selfhost_tinyc_wasm.sh does at its final link). The shim
            # is built lazily by `pixi run build_tinyc_wasm`; if it
            # isn't there, just link without it.
            extras: list[Path] = []
            vararg = ROOT / "tinyc_wasm_vararg.wasm.o"
            if vararg.exists():
                extras.append(vararg)
            link_t, link_err = bench_link(c, objs, out_wasm, args.repeats, extras)
            link_times[c.name] = link_t
            link_errors[c.name] = link_err
            if link_err is not None:
                print(f"  (link)                                                   FAIL  {link_err}")
            else:
                print(f"  {'(link)':55s}  {fmt_ms(link_t)}")
                print(f"  {'TOTAL':55s}  {fmt_s(compile_totals[c.name] + link_t)}\n")

    print_per_file_table(results, errors, configs)
    print_link_and_grand_total(compile_totals, link_times, link_errors, configs)
    if "upstream_full" in compile_totals:
        print_speedup(compile_totals, link_times, configs)

    if args.csv:
        write_csv(args.csv, results, link_times, configs)

    if args.wasmtime:
        compare_produced_wasms(configs)
        wasmtime_bin = shutil.which(args.wasmtime_binary) or args.wasmtime_binary
        if not Path(wasmtime_bin).exists():
            print(
                f"\nerror: wasmtime binary {args.wasmtime_binary!r} not found "
                f"on PATH; install wasmtime or pass --wasmtime-binary",
                file=sys.stderr,
            )
            return 1
        wt_results, wt_errors, wt_link_times, wt_link_errors, wt_compile_totals = (
            run_wasmtime_stage(configs, wasmtime_bin, args.repeats)
        )
        print_per_file_table(wt_results, wt_errors, configs)
        print_link_and_grand_total(wt_compile_totals, wt_link_times, wt_link_errors, configs)
        # Speedup framing: with wasmtime, the most-optimized backend
        # (upstream_full) should produce the fastest compiler. Report
        # ratios relative to native (the slowest produced binary, since
        # our own backend does no optimization).
        if "native" in wt_compile_totals and wt_compile_totals["native"] > 0:
            base_name = "native"
            base_lt = wt_link_times.get(base_name, 0.0)
            base_total = wt_compile_totals[base_name] + (
                0.0 if base_lt == float("inf") else base_lt
            )
            if base_total > 0:
                print(f"\nrelative to {base_name} (wasmtime stage, lower = faster):")
                for c in configs:
                    if c.name == base_name:
                        continue
                    lt = wt_link_times.get(c.name, 0.0)
                    total = wt_compile_totals[c.name] + (
                        0.0 if lt == float("inf") else lt
                    )
                    if total <= 0:
                        continue
                    ratio = total / base_total
                    label = "slower" if ratio >= 1.0 else "faster"
                    magnitude = ratio if ratio >= 1.0 else (base_total / total)
                    print(f"  {c.name:18s} {magnitude:6.2f}x {label}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

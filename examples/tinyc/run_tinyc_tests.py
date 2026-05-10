#!/usr/bin/env python3
"""tinyC end-to-end test runner.

For each test in `tests.toml`:
  - run `./tinyc --emit=llvm tests/X.tc` -> X.ll
  - link with runtime.c using $CC (set by pixi env)
  - run the binary, capture stdout, compare to expected
"""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
IS_WIN = sys.platform == "win32"
TINYC = Path(os.environ.get("TINYC", str(ROOT / ("tinyc.exe" if IS_WIN else "tinyc")))).resolve()
RUNTIME = HERE / "runtime.c"
RUNTIME_WASM = HERE / "runtime_wasm.c"
RUNTIME_WASM_START = HERE / "start_wasm.s"
TESTS_TOML = HERE / "tests.toml"

CC = os.environ.get("CC", "cl" if IS_WIN else "clang")
LLC = os.environ.get("LLC", "llc")
WASM_LD = os.environ.get("WASM_LD", "wasm-ld")
WASMTIME = os.environ.get("WASMTIME", "wasmtime")
WASM_CC = os.environ.get("WASM_CC", "clang")

# Backend used by `tinyc --lowering=...`. Defaults to upstream; override
# with TINYC_LOWERING=native (or any other valid value) to exercise the
# native lowering path through the same suite.
LOWERING = os.environ.get("TINYC_LOWERING", "upstream")

# Code-generation/runtime target for the suite. "native" (default) emits
# LLVM IR via tinyc, then llc + host CC + runtime.c. "wasm" emits a
# wasm32 object via tinyc, then wasm-ld + runtime_wasm.c, and runs the
# resulting .wasm via wasmtime. The wasm target requires
# TINYC_LOWERING=upstream (the native lowering does not implement wasm).
TARGET = os.environ.get("TINYC_TARGET", "native")


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def link_native(obj_path: Path, exe_path: Path):
    """Compile runtime.c and link with the llc-produced object."""
    if IS_WIN:
        # MSVC: cl /nologo /MD obj runtime.c /Fe:exe.exe
        return run([
            CC, "/nologo", "/MD",
            str(obj_path), str(RUNTIME),
            f"/Fe:{exe_path}",
        ])
    cmd = [CC, str(obj_path), str(RUNTIME), "-o", str(exe_path)]
    # llc emits non-PIC by default; some Linux toolchains default to -pie which
    # rejects R_X86_64_32 relocations from .rodata. Force -no-pie on Linux.
    if sys.platform.startswith("linux"):
        cmd.append("-no-pie")
    return run(cmd)


def build_wasm_runtime(out_path: Path, start_obj: Path):
    """Compile runtime_wasm.c (and the _start asm shim) into wasm32
    relocatable object files once per test run. Cached on disk; only
    rebuilt if missing or older than the source."""
    runtime_stale = (
        not out_path.exists()
        or out_path.stat().st_mtime < RUNTIME_WASM.stat().st_mtime
    )
    if runtime_stale:
        r = run([
            WASM_CC, "--target=wasm32-wasi",
            "-nostdlib", "-nostdlibinc", "-fno-builtin", "-O1",
            "-c", str(RUNTIME_WASM), "-o", str(out_path),
        ])
        if r.returncode != 0:
            return r
    start_stale = (
        not start_obj.exists()
        or start_obj.stat().st_mtime < RUNTIME_WASM_START.stat().st_mtime
    )
    if start_stale:
        r = run([
            WASM_CC, "--target=wasm32",
            "-c", str(RUNTIME_WASM_START), "-o", str(start_obj),
        ])
        if r.returncode != 0:
            return r
    return None


def link_wasm(obj_path: Path, runtime_obj: Path, start_obj: Path,
              wasm_path: Path):
    """Link a tinyc-emitted wasm32 object together with the wasm runtime
    + _start shim using wasm-ld, producing a runnable .wasm module."""
    return run([
        WASM_LD, "--no-entry", "--export=_start",
        str(obj_path), str(runtime_obj), str(start_obj),
        "-o", str(wasm_path),
    ])


def main():
    if not TINYC.exists():
        print(f"error: {TINYC} not found; run build_tinyc_upstream first", file=sys.stderr)
        return 2
    cfg = tomllib.loads(TESTS_TOML.read_text())
    tests = cfg.get("test", [])
    if not tests:
        print("error: no tests found in tests.toml", file=sys.stderr)
        return 2

    # Platform key matching `platforms = [...]` in tests.toml. A test
    # with a `platforms` field runs only on listed platforms. Values:
    # "linux", "darwin", "win32".
    plat = sys.platform if sys.platform.startswith(("linux", "darwin", "win32")) else sys.platform
    if plat.startswith("linux"):
        plat_key = "linux"
    elif plat.startswith("darwin"):
        plat_key = "darwin"
    elif plat.startswith("win32"):
        plat_key = "win32"
    else:
        plat_key = plat

    # The native lowering is allowed with the wasm target: the upstream
    # `MLIR_TranslateModuleToWasm` is responsible for the LLVM->wasm step,
    # and the lowering choice only affects how MLIR is reduced to the LLVM
    # dialect beforehand.

    # Pre-build the wasm runtime object once per run.
    wasm_runtime_obj = HERE / "tests" / "runtime_wasm.o"
    wasm_start_obj   = HERE / "tests" / "start_wasm.o"
    if TARGET == "wasm":
        r = build_wasm_runtime(wasm_runtime_obj, wasm_start_obj)
        if r is not None and r.returncode != 0:
            print(f"error: failed to compile wasm runtime\nstderr:\n{r.stderr}",
                  file=sys.stderr)
            return 2

    failures = 0
    skipped = 0
    for t in tests:
        name = t["name"]
        expected = t["expected_stdout"]
        platforms = t.get("platforms")
        if platforms is not None and plat_key not in platforms:
            print(f"SKIP {name} (platforms={platforms}, current={plat_key})")
            skipped += 1
            continue
        # Optional opt-out for the wasm target only (e.g. tests that
        # rely on host-libc behavior wasm32-wasi can't reproduce).
        if TARGET == "wasm" and t.get("wasm_skip"):
            print(f"SKIP {name} (wasm_skip)")
            skipped += 1
            continue
        # Multi-file tests pass `sources = [...]`; single-file tests
        # default to `<name>.tc` for backwards compatibility.
        sources = t.get("sources", [f"{name}.tc"])
        srcs = [HERE / "tests" / s for s in sources]

        if TARGET == "wasm":
            obj  = HERE / "tests" / f"{name}.wasm.o"
            wasm = HERE / "tests" / f"{name}.wasm"

            # Stage 1: tinyc emits wasm32 object directly.
            r = run([str(TINYC), "--emit=wasm", f"--lowering={LOWERING}",
                     "-I", str(HERE / "tests"),
                     "-o", str(obj),
                     *[str(s) for s in srcs]])
            if r.returncode != 0:
                print(f"FAIL {name}: tinyc returned {r.returncode}\nstderr:\n{r.stderr}")
                failures += 1
                continue

            # Stage 2: link with wasm-ld + runtime_wasm.o + start_wasm.o.
            r = link_wasm(obj, wasm_runtime_obj, wasm_start_obj, wasm)
            if r.returncode != 0:
                print(f"FAIL {name}: wasm-ld failed\nstderr:\n{r.stderr}\nstdout:\n{r.stdout}")
                failures += 1
                continue

            # Stage 3: run under wasmtime.
            r = run([WASMTIME, str(wasm)])
            if r.returncode != 0:
                print(f"FAIL {name}: wasm exited with status {r.returncode}\nstdout: {r.stdout!r}\nstderr: {r.stderr!r}")
                failures += 1
                continue
            if r.stdout != expected:
                print(f"FAIL {name}: stdout mismatch\n  expected: {expected!r}\n  got:      {r.stdout!r}")
                failures += 1
                continue

            print(f"PASS {name}")
            continue

        ll  = HERE / "tests" / f"{name}.ll"
        obj = HERE / "tests" / (f"{name}.obj" if IS_WIN else f"{name}.o")
        exe = HERE / "tests" / (f"{name}.exe" if IS_WIN else f"{name}.bin")

        # Stage 1: emit LLVM IR. The driver accepts multiple input files
        # and merges them into a single MLIR module.
        # `-I tests` so multi-file tests can `#include "shared.h"`.
        r = run([str(TINYC), "--emit=llvm", f"--lowering={LOWERING}",
                 "-I", str(HERE / "tests"),
                 *[str(s) for s in srcs]])
        if r.returncode != 0:
            print(f"FAIL {name}: tinyc returned {r.returncode}\nstderr:\n{r.stderr}\nstdout (first 200 chars):\n{r.stdout[:200]}")
            failures += 1
            continue
        if not r.stdout.strip():
            print(f"FAIL {name}: tinyc produced empty LLVM IR\nstderr:\n{r.stderr}")
            failures += 1
            continue
        ll.write_text(r.stdout)

        # Stage 2: compile .ll -> .o/.obj with llc (works regardless of $CC),
        # then link with the platform compiler + runtime.c.
        r = run([LLC, "-filetype=obj", str(ll), "-o", str(obj)])
        if r.returncode != 0:
            print(f"FAIL {name}: llc failed\nstderr:\n{r.stderr}")
            failures += 1
            continue
        r = link_native(obj, exe)
        if r.returncode != 0:
            print(f"FAIL {name}: link failed\nstderr:\n{r.stderr}\nstdout:\n{r.stdout}")
            failures += 1
            continue

        # Stage 3: run
        r = run([str(exe)])
        if r.returncode != 0:
            print(f"FAIL {name}: binary exited with status {r.returncode}\nstdout: {r.stdout!r}\nstderr: {r.stderr!r}")
            failures += 1
            continue
        if r.stdout != expected:
            print(f"FAIL {name}: stdout mismatch\n  expected: {expected!r}\n  got:      {r.stdout!r}")
            failures += 1
            continue

        print(f"PASS {name}")

    if failures:
        print(f"\n{failures} test(s) failed")
        return 1
    ran = len(tests) - skipped
    if skipped:
        print(f"\nAll {ran} tinyC tests passed ({skipped} skipped on {plat_key})")
    else:
        print(f"\nAll {len(tests)} tinyC tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

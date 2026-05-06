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
TINYC = ROOT / ("tinyc.exe" if IS_WIN else "tinyc")
RUNTIME = HERE / "runtime.c"
TESTS_TOML = HERE / "tests.toml"

CC = os.environ.get("CC", "cl" if IS_WIN else "clang")
LLC = os.environ.get("LLC", "llc")


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


def main():
    if not TINYC.exists():
        print(f"error: {TINYC} not found; run build_tinyc_upstream first", file=sys.stderr)
        return 2
    cfg = tomllib.loads(TESTS_TOML.read_text())
    tests = cfg.get("test", [])
    if not tests:
        print("error: no tests found in tests.toml", file=sys.stderr)
        return 2

    failures = 0
    for t in tests:
        name = t["name"]
        expected = t["expected_stdout"]
        src = HERE / "tests" / f"{name}.tc"
        ll  = HERE / "tests" / f"{name}.ll"
        obj = HERE / "tests" / (f"{name}.obj" if IS_WIN else f"{name}.o")
        exe = HERE / "tests" / (f"{name}.exe" if IS_WIN else f"{name}.bin")

        # Stage 1: emit LLVM IR
        r = run([str(TINYC), "--emit=llvm", str(src)])
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
    print(f"\nAll {len(tests)} tinyC tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

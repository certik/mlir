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
TINYC = ROOT / "tinyc"
RUNTIME = HERE / "runtime.c"
TESTS_TOML = HERE / "tests.toml"

CC = os.environ.get("CC", "clang")


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


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
        exe = HERE / "tests" / f"{name}.bin"

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

        # Stage 2: link
        r = run([CC, str(ll), str(RUNTIME), "-o", str(exe)])
        if r.returncode != 0:
            print(f"FAIL {name}: link failed\n{r.stderr}")
            failures += 1
            continue

        # Stage 3: run
        r = run([str(exe)])
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

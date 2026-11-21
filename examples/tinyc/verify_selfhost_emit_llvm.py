#!/usr/bin/env python3
"""
CI verification: a self-hosted tinyc must print the same LLVM IR as the
clang-built one.

The self-host fixed-point check (stage 2 == stage 3) cannot catch bugs in
the runtime the self-hosted compiler links (corec + corec-stdlib instead of
the host libc), because stage 2 and stage 3 share them. For example, a
`vsnprintf(NULL, 0, ...)` that returned 0 made every integer literal empty
in `--emit=llvm` output (#322). This script runs `--emit=llvm` on every
tests/*.tc with both compilers and requires byte-identical output.

Usage:
    python examples/tinyc/verify_selfhost_emit_llvm.py REF_TINYC SELFHOST_TINYC
e.g. ./tinyc_native ./tinyc_stage2_elf
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# The include path run_tinyc_tests.py compiles tests with, so tests that
# include <stdio.h> etc. (corec-stdlib) are compared too.
INCLUDES = [
    "-I", str(ROOT / "corec-stdlib" / "stdlib"),
    "-I", str(ROOT / "corec-stdlib" / "corec"),
    "-I", str(HERE / "tests"),
]

# builtin_syscall.tc: driver.c's inject_syscall6_definition() only defines
# __tinyc_syscall6 when tinyc itself is compiled with
# `defined(__linux__) && defined(__x86_64__) && !defined(__TINYC__)`, which
# a self-hosted tinyc is not (#326).
SKIP = {"builtin_syscall": "#326: no __tinyc_syscall6 definition when self-hosted"}


def emit(tinyc: str, src: Path, out: Path) -> bool:
    out.unlink(missing_ok=True)
    proc = subprocess.run([tinyc, "--emit=llvm", *INCLUDES, str(src), "-o", str(out)],
                          capture_output=True)
    return proc.returncode == 0 and out.exists()


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_selfhost_emit_llvm.py REF_TINYC SELFHOST_TINYC",
              file=sys.stderr)
        return 2
    ref, selfhost = (os.path.abspath(a) for a in sys.argv[1:])
    checked = 0
    failed = []
    skipped = []
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "ref.ll", Path(tmp) / "selfhost.ll"
        for src in sorted((HERE / "tests").glob("*.tc")):
            if src.stem in SKIP:
                skipped.append(f"{src.name} ({SKIP[src.stem]})")
                continue
            if not emit(ref, src, a):
                # e.g. one TU of a multi-file test that needs the others
                skipped.append(f"{src.name} (reference compiler fails on it alone)")
                continue
            checked += 1
            if not emit(selfhost, src, b) or a.read_bytes() != b.read_bytes():
                failed.append(src.name)
    for name in skipped:
        print(f"skipped: {name}")
    for name in failed:
        print(f"MISMATCH: {name}")
    print(f"--emit=llvm: {checked - len(failed)}/{checked} identical "
          f"({os.path.basename(selfhost)} vs {os.path.basename(ref)})")
    return 1 if failed or checked == 0 else 0


if __name__ == "__main__":
    sys.exit(main())

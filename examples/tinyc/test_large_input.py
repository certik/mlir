#!/usr/bin/env python3
"""Compile one large generated C file to wasm and check that it succeeds.

A regression test for #354/#373: the native backend reserves its IR vectors
from a size hint, and an over-estimate for `--emit=wasm` once used up corec's
4 GiB heap on inputs the pointer storage compiled (140K one-line functions,
about 5 MB). Too slow for CI (about 70 s with the checked `tinyc_native`), so
it is the manual task `pixi run -e upstream test_tinyc_large_input`.

usage: test_large_input.py [TINYC] [N_FUNCTIONS]
"""
import os
import subprocess
import sys
import tempfile
import time

tinyc = sys.argv[1] if len(sys.argv) > 1 else "./tinyc_native"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 140000

with tempfile.TemporaryDirectory() as d:
    src = os.path.join(d, "large.c")
    with open(src, "w") as f:
        for k in range(n):
            f.write(f"int f{k}(int a){{return a+{k};}}\n")
        f.write("int main(void){return f0(1);}\n")
    t0 = time.time()
    r = subprocess.run([tinyc, "--emit=wasm", "--lowering=native",
                        "-o", os.path.join(d, "large.wasm.o"), src],
                       capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        print(f"FAIL: tinyc --emit=wasm of {n} functions returned {r.returncode} "
              f"after {dt:.1f} s\n{r.stderr[-2000:]}")
        sys.exit(1)
    print(f"PASS: {n} functions compiled to wasm in {dt:.1f} s")

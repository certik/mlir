#!/usr/bin/env python3
"""Build the tinyC compiler itself as a wasm32-wasi module.

Output is ``tinyc.wasm`` -- a self-contained ``.wasm`` that, given a tinyC
source on its argv (via the WASI virtual FS), produces a wasm32 object on
stdout or ``-o <path>``. Combined with ``--link``, the same binary also
acts as the linker. This is what the browser demo loads.

Also pre-builds the runtime/start/vararg objects shipped alongside
``tinyc.wasm`` and used by ``tinyc --link``.

This is a Python driver (rather than a shell script) so it runs unchanged
on Linux, macOS, and Windows. pixi runs tasks through deno_task_shell
(the ``shell`` package), whose version differs by platform -- linux-64
gets 0.2.0 from conda-forge while win-64/osx-arm64 get 0.3.0 -- and the
older parser rejects clang flags such as ``-Wl,-z,stack-size=...``. Driving
clang from Python sidesteps that shell-version skew entirely.
"""

import subprocess
import sys

# Source set. Keep in sync with the source list in `selfhost_tinyc_wasm.py`.
SOURCES = [
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
    "corec-stdlib/stdlib/stdio.c",
    "corec-stdlib/stdlib/stdlib.c",
    "corec-stdlib/stdlib/printf.c",
    "corec-stdlib/stdlib/string_impl.c",
    "examples/tinyc/lex.c",
    "examples/tinyc/preprocess.c",
    "examples/tinyc/parse.c",
    "examples/tinyc/emit.c",
    "examples/tinyc/driver.c",
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
    "mlir_wasm_to_wasmstack.c",
    "mlir_wasmstack_to_wasmssa.c",
    "mlir_wasmssa_to_llvm.c",
    "mlir_llvm_mem2reg.c",
    "mlir_llvm_load_cse.c",
    "mlir_llvm_arith_gvn.c",
    "mlir_llvm_dce.c",
    "mlir_regalloc.c",
    "mlir_llvm_to_aarch64.c",
    "mlir_aarch64_asm.c",
    "mlir_aarch64_to_macho.c",
    "tokenizer.c",
    "mlir_parser.c",
    "op_parsers.c",
    "mlir_classic_printer.c",
    "mlir_generic_printer.c",
    "mlir_lift_cf_to_scf.c",
    "corec/platform/platform_wasm.c",
]


def run(cmd):
    print(" ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.stderr.write("error: command failed: " + " ".join(cmd) + "\n")
        sys.exit(proc.returncode)


def main():
    # Build the single self-contained tinyc.wasm (compiler + linker).
    run(
        ["clang",
         "--target=wasm32-wasi",
         "-Os",
         "-Wl,-z,stack-size=4194304",
         "-nostdlib",
         "-nostdinc",
         "-fno-builtin",
         "-I", "corec",
         "-I", "corec-stdlib/stdlib",
         "-I", ".",
         "-Wl,--no-entry",
         "-Wl,--export=_start",
         "-Wl,--export=wasm_buddy_alloc",
         "-Wl,--export=wasm_buddy_free",
         "-Wl,--initial-memory=33554432",
         "-o", "tinyc.wasm"]
        + SOURCES
    )

    # Pre-build the runtime + start objects we ship alongside tinyc.wasm so
    # the browser demo can hand them to `tinyc --link` without invoking
    # clang in the browser.
    run(
        ["clang", "--target=wasm32-wasi", "-O2", "-nostdinc", "-fno-builtin",
         "-I", "corec", "-I", "corec-stdlib/stdlib",
         "-c", "-o", "runtime_wasm.wasm.o", "examples/tinyc/runtime_wasm.c"]
    )

    run(
        ["clang", "--target=wasm32", "-O2", "-nostdlib", "-fno-builtin",
         "-c", "-o", "start_wasm.wasm.o", "examples/tinyc/start_wasm.s"]
    )

    # Minimal `tinyc_va_arg_*` shim used by tinyC-compiled wasm binaries
    # (selfhost stage 2+, or any program tinyC compiles that uses varargs
    # in a non-browser context). Kept separate from runtime_wasm.wasm.o so
    # it can be linked alongside a tinyC-compiled corec-stdlib without
    # duplicating printf/strlen/etc.
    run(
        ["clang", "--target=wasm32-wasi", "-O2", "-nostdinc", "-fno-builtin",
         "-I", "corec", "-I", "corec-stdlib/stdlib",
         "-c", "-o", "tinyc_wasm_vararg.wasm.o",
         "examples/tinyc/tinyc_wasm_vararg.c"]
    )

    import os
    for f in ["tinyc.wasm", "runtime_wasm.wasm.o", "start_wasm.wasm.o",
              "tinyc_wasm_vararg.wasm.o"]:
        print("%10d  %s" % (os.path.getsize(f), f))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Self-host the tinyC compiler.

Use an existing ``tinyc.wasm`` (under wasmtime) to recompile every source
file that makes up the tinyc wasm32-wasi build into ``.wasm.o`` objects,
then use that same ``tinyc.wasm`` to link them into a fresh ``tinyc.wasm``.

Usage:
    selfhost_tinyc_wasm.py <INPUT_TINYC_WASM> <OUTPUT_TINYC_WASM> <STAGE_DIR>

This is a Python driver (rather than a shell script) so it runs unchanged
on Linux, macOS, and Windows. The ``shell`` interpreter pixi uses on
Windows (deno_task_shell) does not support arrays, ``set --``, or
word-splitting of variables, which a compile-each-then-link-all loop needs.

The source set mirrors ``build_tinyc_wasm.py`` exactly so the stage-2 and
stage-3 outputs can be compared byte-for-byte to verify self-hosting
reproducibility. Keep these two lists in sync.
"""

import os
import subprocess
import sys

# Source set mirrors `build_tinyc_wasm.py`. Keep these in sync.
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
]
PLATFORM_C_FILES = [
    "corec/platform/platform_wasm.c",
]

ALL_SOURCES = (
    COREC_C_FILES
    + COREC_STDLIB_C_FILES
    + TINYC_C_FILES
    + NATIVE_C_FILES
    + PLATFORM_C_FILES
)

INCLUDES = ["-I", "corec", "-I", "corec-stdlib/stdlib", "-I", "."]


def run(cmd):
    """Run a command, echoing it on failure, and abort on non-zero exit."""
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.stderr.write("error: command failed: " + " ".join(cmd) + "\n")
        sys.exit(proc.returncode)


def main():
    if len(sys.argv) != 4:
        sys.stderr.write(
            "usage: %s INPUT_TINYC_WASM OUTPUT_TINYC_WASM STAGE_DIR\n"
            % sys.argv[0]
        )
        sys.exit(2)

    input_wasm, output_wasm, stage_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    if not os.path.isfile(input_wasm):
        sys.stderr.write("error: input compiler not found: %s\n" % input_wasm)
        sys.exit(1)

    os.makedirs(stage_dir, exist_ok=True)

    # tinyC lowers va_arg inline (the portable 8-byte cursor model) and `_start`
    # comes from tinyC-compiled `corec/platform/platform_wasm.c`, so the
    # self-hosted wasm needs no external support objects at link time.
    objs = []
    for src in ALL_SOURCES:
        obj = "%s/%s.wasm.o" % (stage_dir, src.replace("/", "_"))
        print("[tinyc.wasm] %s -> %s" % (src, obj))
        run(
            ["wasmtime", "--dir", ".", "--dir", stage_dir, input_wasm,
             "--emit=wasm", "--lowering=native"]
            + INCLUDES
            + ["-o", obj, src]
        )
        objs.append(obj)

    print("[tinyc.wasm] --link -> %s" % output_wasm)
    run(
        ["wasmtime", "--dir", ".", "--dir", stage_dir, input_wasm, "--link",
         "--export=_start", "-o", output_wasm]
        + objs
    )

    print("wrote %s" % output_wasm)


if __name__ == "__main__":
    main()

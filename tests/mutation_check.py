#!/usr/bin/env python3
"""Mutation sanity check for the test infrastructure.

The unified test runner only catches regressions if a real bug in the
parser/printer code paths actually breaks at least one test. This script
injects a series of one-line mutations into representative code paths,
rebuilds, runs the relevant test suite, and asserts that the suite
fails. If any mutation leaves the test suite green, that's a coverage
hole and the mutation list must grow.

Each mutation is reverted before the next one is tried. After the run,
the tree is left in its original state and rebuilt clean.

Usage:
    pixi r --environment upstream python tests/mutation_check.py

Exit status 0 when every mutation broke the expected suite, non-zero
when at least one mutation slipped through (i.e. coverage gap).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Mutation:
    name: str
    file: str          # path relative to repo root
    old: str           # exact substring to find (must be unique)
    new: str           # replacement
    suite: str         # "tests" or "validate-refs"
    rebuild: tuple     # tasks to run before the suite, e.g. ("build_macos",)
    regen_refs: bool = False  # regenerate refs against mutated output before
                              # running the suite (used to demonstrate that
                              # validate-refs catches things `tests` does not)


# Pick one mutation per code path. Keep them tiny but visible-in-output
# so the resulting diff is unmistakable. Each must be unique in its file
# so the find/replace is unambiguous.
MUTATIONS: list[Mutation] = [
    # --- native classic printer ---
    Mutation(
        name="classic-printer:return-keyword",
        file="mlir_classic_printer.c",
        old='result = str_concat(arena, result, str_lit("return"));',
        new='result = str_concat(arena, result, str_lit("returnX"));',
        suite="tests",
        rebuild=("build_macos",),
    ),
    Mutation(
        name="classic-printer:arith.constant-keyword",
        file="mlir_classic_printer.c",
        old='result = str_concat(arena, result, str_lit("arith.constant "));',
        new='result = str_concat(arena, result, str_lit("arith.constantX "));',
        suite="tests",
        rebuild=("build_macos",),
    ),

    # --- native generic printer (opname-quote opener) ---
    Mutation(
        name="generic-printer:opname-quote-open",
        file="mlir_generic_printer.c",
        # First "\"" emission opens the op-name quote; flip it so all
        # generic refs come out unquoted and parsing them fails.
        old='        result = str_concat(arena, result, str_lit("\\""));\n        if (opname.size > 0) result = str_concat(arena, result, opname);',
        new='        result = str_concat(arena, result, str_lit("X"));\n        if (opname.size > 0) result = str_concat(arena, result, opname);',
        suite="tests",
        rebuild=("build_macos",),
    ),

    # --- native parser (op-name dispatch) ---
    Mutation(
        name="parser:func.return-dispatch",
        file="mlir_parser.c",
        old='} else if (str_eq(opname, str_lit("func.return"))) {',
        new='} else if (str_eq(opname, str_lit("func.returnX"))) {',
        suite="tests",
        rebuild=("build_macos",),
    ),

    # --- validate-refs guardrail ---
    # The first 4 mutations all break the regular `tests` suite, which is
    # easy to catch. To prove that `--validate-refs` adds independent
    # signal, we mutate the affine.load typed printer so its output is
    # syntactically broken (extra junk before the source operand), then
    # regenerate refs against that mutated printer. The regular test
    # suite passes because the refs match the printer; only
    # --validate-refs catches that the refs no longer round-trip through
    # upstream's parser.
    Mutation(
        name="validate-refs:affine.load-typed-form",
        file="mlir_classic_printer.c",
        old='case OP_TYPE_AFFINE_LOAD: {\n            // Typed form: %r = affine.load %memref[%i, ...] : memref<...>',
        new='case OP_TYPE_AFFINE_LOAD: {\n            result = str_concat(arena, result, str_lit("@@invalid@@ "));\n            // Typed form: %r = affine.load %memref[%i, ...] : memref<...>',
        suite="validate-refs",
        rebuild=("build_macos", "build_parser_upstream"),
        regen_refs=True,
    ),
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO, **kw)


def pixi(*args: str) -> subprocess.CompletedProcess:
    return run(["pixi", "r", *args], capture_output=True, text=True)


def pixi_env(env: str, *args: str) -> subprocess.CompletedProcess:
    return run(["pixi", "r", "--environment", env, *args],
               capture_output=True, text=True)


def apply_mutation(m: Mutation) -> str:
    p = REPO / m.file
    original = p.read_text()
    if m.old not in original:
        raise SystemExit(f"[{m.name}] anchor not found in {m.file}: {m.old!r}")
    if original.count(m.old) != 1:
        raise SystemExit(f"[{m.name}] anchor not unique in {m.file}")
    p.write_text(original.replace(m.old, m.new))
    return original


def restore(file: str, original: str) -> None:
    (REPO / file).write_text(original)


def rebuild(tasks: tuple[str, ...]) -> tuple[bool, str]:
    for t in tasks:
        if t == "build_parser_upstream":
            r = pixi_env("upstream", "build_parser_upstream")
        elif t == "build_macos":
            r = pixi_env("macos", "build_macos")
        else:
            r = pixi(t)
        if r.returncode != 0:
            # A build failure also counts as a "test broke" because it
            # means the mutation prevents the binary from being usable.
            return False, (r.stdout + r.stderr)[-400:]
    return True, ""


def run_suite(suite: str) -> int:
    if suite == "tests":
        r = pixi_env("upstream", "python", "run_tests.py")
    elif suite == "validate-refs":
        r = pixi_env("upstream", "python", "run_tests.py", "--validate-refs")
    else:
        raise SystemExit(f"unknown suite {suite}")
    return r.returncode


def main() -> int:
    print(f"Running {len(MUTATIONS)} mutations...\n")
    failures: list[str] = []   # mutations that did NOT break the suite
    for m in MUTATIONS:
        print(f"[{m.name}] applying...", flush=True)
        original = apply_mutation(m)
        refs_backup: str | None = None
        try:
            built, err = rebuild(m.rebuild)
            if not built:
                print(f"  build failed (counts as caught) ✓\n    err tail: {err!r}")
                continue
            if m.regen_refs:
                refs_backup = tempfile.mkdtemp(prefix="mut_refs_")
                shutil.copytree(REPO / "tests/reference", Path(refs_backup) / "ref",
                                dirs_exist_ok=True)
                # Regenerate refs against the mutated printer so the
                # regular tests would pass — only --validate-refs should
                # surface the breakage.
                pixi_env("upstream", "python", "run_tests.py", "-u")
                rc_tests = run_suite("tests")
                if rc_tests != 0:
                    print(f"  unexpected: regular tests still fail after -u ✗")
                    failures.append(m.name + " (tests-after-regen)")
                    continue
            rc = run_suite(m.suite)
            if rc == 0:
                print(f"  suite '{m.suite}' STILL PASSED — coverage gap! ✗")
                failures.append(m.name)
            else:
                print(f"  suite '{m.suite}' failed as expected ✓")
        finally:
            restore(m.file, original)
            if refs_backup is not None:
                # Restore refs from backup so the tree stays clean.
                shutil.rmtree(REPO / "tests/reference")
                shutil.copytree(Path(refs_backup) / "ref",
                                REPO / "tests/reference")
                shutil.rmtree(refs_backup)

    # Final clean rebuild so the tree is left usable.
    print("\nRestoring clean build...")
    pixi_env("macos", "build_macos")
    pixi_env("upstream", "build_parser_upstream")

    print()
    if failures:
        print("COVERAGE GAPS:")
        for n in failures:
            print(f"  - {n}")
        return 1
    print(f"All {len(MUTATIONS)} mutations were caught by the test infra.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

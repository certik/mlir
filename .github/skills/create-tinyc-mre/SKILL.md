---
name: create-tinyc-mre
description: >
  Create a Minimal Reproducible Example (MRE) for a tinyC compiler bug as a
  test under examples/tinyc/tests/. Use this skill when C code compiles and
  runs correctly with clang but fails to compile, miscompiles, or produces
  wrong output with tinyC (examples/tinyc). Produces a tests/<name>.tc file
  plus an entry in tests.toml. Triggers: tinyc MRE, minimal reproducible
  example, reduce, reproduce, reproducer, tinyc bug, compilation error,
  wrong output, miscompile, bug reduction.
---

# Create tinyC MRE — Minimal Reproducible Example as a Test

Reduce a tinyC compiler failure into the smallest possible standalone C program
that reproduces the bug, and add it as a test in `examples/tinyc/tests/`. The
MRE must compile and run correctly with `clang` (the reference) but fail with
`./tinyc` — either at compile time, link time, or by producing different stdout
when run.

The goal is **not just to reproduce** but to land a regression test that will
PASS once the bug is fixed. The same test serves both as the bug report and
as the regression guard.

## Prerequisites

Before starting, confirm:

- The tinyC binary `./tinyc` exists at the repo root. If not, build it:
  `pixi run -e upstream build_tinyc_upstream`
- `clang` is available on PATH (it ships with macOS / pixi env).
- You have the failing C source — either a snippet, a preprocessed `.i` file,
  or a path inside `corec/`, `corec-stdlib/`, or this repo's own sources.
- You know the exact tinyC error or wrong-output symptom.

## Inputs to Gather

Ask the user for (if not already provided):

1. **The failing source** — file path, `.i` file, or pasted snippet.
2. **The exact error or symptom** — copy the `tinyc parse error at line ...`,
   `tinyc emit error ...`, `lowering error`, link error, or stdout diff.
3. **Failure type** — pick one:
   - **Parse error**: `tinyc parse error ...`
   - **Emit error**: `tinyc emit error ...`
   - **Lowering error**: MLIR pass-pipeline failure (e.g. `out of bounds`)
   - **Link error**: undefined symbol
   - **Wrong output**: tinyc-built binary runs but prints wrong stdout
4. **A short bug name** — kebab-or-snake, used as the test filename
   (e.g. `union_field_offset`, `fnptr_struct_return`, `ptr_postfix_stride`).

## Procedure

### Phase 1: Reproduce the Failure

1. If the input is a preprocessed `.i` file, run tinyc directly:
   ```bash
   ./tinyc --emit=llvm -o /tmp/x.ll path/to/input.i
   ```
2. If the input is a normal `.c` file, preprocess first:
   ```bash
   clang -E -P -nostdinc -fno-builtin -DNDEBUG \
     -I corec -I corec-stdlib/stdlib -I . \
     path/to/input.c -o /tmp/x.i
   ./tinyc --emit=llvm -o /tmp/x.ll /tmp/x.i
   ```
3. Capture the exact tinyc output (error message, line number, op name).
   This is the **failure signature** — every reduction step must preserve it.

### Phase 2: Isolate the Failing Construct

Identify the C feature implicated by the error:

| Symptom | Likely construct |
|---|---|
| `expected function name` after `print` / common identifier | builtin name collision |
| `out of bounds` GEP in lowering | struct/union layout, array stride |
| `undefined function` | `extern` declaration / linkage / variadics |
| miscompile reading wrong bytes | union aliasing, ptr stride, field offset |
| `expected type` in declarator | typedef/fnptr/struct grammar gap |

Find the smallest containing scope (function, struct definition, expression)
that still triggers the error.

### Phase 3: Reduce via Binary Search

Starting from the isolated unit:

1. Strip everything not referenced by the failing construct: unrelated funcs,
   unused globals, unused includes.
2. Replace each remaining call to corec / corec-stdlib with a hand-written
   stub. **The MRE must be self-contained** — only `examples/tinyc/runtime.h`
   helpers like `_tinyc_print(int)` are available without a stub.
3. Inline typedefs; collapse multi-step macros.
4. After each removal, re-run:
   ```bash
   clang -O0 -o /tmp/ref /tmp/mre.c   # must still compile + run + give same output
   ./tinyc --emit=llvm -o /tmp/x.ll /tmp/mre.c
   ```
   Both must still exhibit the failure signature; the clang reference must
   still produce the expected output.
5. Stop when nothing else can be removed without losing the bug.

### Phase 4: Restructure as a tinyC Test

The test format is:

```c
// Brief description of the bug and what semantics the test guards.

int main(void) {
    // ... exercise the construct ...
    if (/* bad result */) { _tinyc_print(1); return 1; }
    if (/* bad result */) { _tinyc_print(2); return 2; }
    // ... more checks ...
    _tinyc_print(0);
    return 0;
}
```

Conventions (verified against existing tests in
`examples/tinyc/tests/struct_zero_init_64bit.tc`,
`examples/tinyc/tests/union_field_offset.tc`,
`examples/tinyc/tests/ptr_postfix_stride.tc`):

- **No `#include`**. tinyC has no preprocessor for files; declare what you
  need locally (`extern int printf(const char *, ...);` is fine).
- **Use `_tinyc_print(int)`** for output (NOT `print()` — it was renamed in
  PR #124). It prints the integer + newline. There's also `_tinyc_print` for
  float and string.
- **Print 0 and return 0 on success**, print a non-zero index on each
  specific failure path so the test reports *which* check failed.
- **Self-contained**: define every type and helper inside the file. Don't
  pull in corec headers.
- **Minimum size**: typically 20-80 lines.

### Phase 5: Wire Up the Test

1. Save the MRE as `examples/tinyc/tests/<name>.tc`.
2. Append an entry to `examples/tinyc/tests.toml`:
   ```toml
   [[test]]
   name = "<name>"
   expected_stdout = "0\n"
   ```
   The `expected_stdout` is whatever `_tinyc_print(0)` (or other final
   print) emits when the test passes. Get this from the clang reference
   binary you already verified in Phase 3.

### Phase 6: Verify

Run BOTH checks:

```bash
# 1. Reference: clang must compile and produce the expected output.
clang -O0 -o /tmp/mre_ref examples/tinyc/tests/<name>.tc \
      examples/tinyc/runtime.c
/tmp/mre_ref
# Must print exactly the expected_stdout you put in tests.toml.

# 2. Bug: tinyc must reproduce the failure (until the underlying bug is
#    fixed). If it's a parse/emit/lowering error, the test will appear in
#    the FAIL list of test_tinyc_upstream. If it's a wrong-output bug, the
#    diff will show in the FAIL line.
pixi run -e upstream test_tinyc_upstream 2>&1 | grep -E "<name>|tests passed"
```

For a **bug not yet fixed**, the test SHOULD fail. That's the point — it
documents the bug and gates the fix. Once the bug is fixed it will start
passing automatically (CI catches regressions).

For a **bug being fixed in the same change**, run the same command and
verify the test now passes alongside the others (`All N tinyC tests passed`).

### Phase 7: Summary

Print to the user:

```
tinyC MRE created.

Files:
  examples/tinyc/tests/<name>.tc  (NEW)
  examples/tinyc/tests.toml       (entry added)

Bug: <one-line description>
Failure signature: <copy the key line from tinyc output>
Status: <FAILS | PASSES> in test_tinyc_upstream

Reproduce:
  pixi run -e upstream build_tinyc_upstream
  ./tinyc --emit=llvm -o /tmp/x.ll examples/tinyc/tests/<name>.tc

Verify with clang:
  clang -O0 -o /tmp/ref examples/tinyc/tests/<name>.tc \
        examples/tinyc/runtime.c && /tmp/ref
```

## Tips

- **Bootstrap-driven bugs**: When `pixi run -e macos test_macos_tinyc` fails,
  the failing source is one of `parser.c`, `mlir_*.c`, `op_parsers.c`,
  `corec/base/*.c`, or `corec-stdlib/stdlib/*.c`. The build script writes the
  preprocessed output to `build_tinyc/<name>.c.i`. Use that as the starting
  point — the construct that crashes tinyc is somewhere in there.
- **Locate the failing line**: `tinyc emit error at line N` refers to the
  LINE IN THE PREPROCESSED `.i` FILE, not the original source. Open the
  `.i` file and look at line N.
- **MLIR lowering errors**: Use `--emit=mlir` instead of `--emit=llvm` to see
  the MLIR before lowering. Run it through `pixi run -e upstream mlir-opt
  --convert-to-llvm` to pinpoint the failing op.
- **Multi-file bugs**: tinyC tests are single-file. If the bug genuinely
  needs cross-translation-unit linkage, use `extern` declarations and inline
  the called function as a static helper.
- **Don't reduce TOO far**: The MRE must trigger the SAME tinyc error
  message, not a different one. After each cut, re-check the failure
  signature is preserved.
- **Naming**: pick a name that describes the C construct, not the MLIR symptom.
  Good: `union_field_offset`, `ptr_postfix_stride`, `fnptr_struct_return`.
  Bad: `gep_oob`, `bug_42`.

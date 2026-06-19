---
name: fix-tinyc-mre
description: >
  Fix a tinyC compiler bug given an MRE that has already been committed by
  the create-tinyc-mre skill. Reads the MRE from HEAD (must be a commit with
  subject "tinyc-mre: <name>"), reproduces the failure, fixes the bug in
  the corec submodule (tinyc source), reruns the MRE plus the full tinyC
  test suite, and commits the fix. Triggers: fix tinyc bug, fix tinyc MRE,
  fix MRE, debug tinyc, patch tinyc, resolve tinyc compile error,
  resolve miscompile.
---

# Fix tinyC MRE — Fix a tinyC Compiler Bug

Given a committed MRE (produced by `create-tinyc-mre`), reproduce the bug,
fix it in the tinyC source (`corec/examples/tinyc/`), confirm the MRE now
passes, run the full tinyC test suite, and commit.

## Prerequisites

- Current working directory is the `mlir` repo root.
- `corec/` is a submodule containing tinyC source. The fix lives there.
- `./tinyc` is built (or can be rebuilt with `pixi run -e upstream
  build_tinyc_upstream`).
- `clang` is available on PATH.

## Hard Precondition: HEAD Must Be a tinyc-mre Commit

This skill **only** operates on a committed MRE. If HEAD is not a
`tinyc-mre:` commit, **stop immediately** and instruct the user to invoke
`create-tinyc-mre` first. Do not attempt to find an MRE elsewhere — there
is exactly one source of truth.

```bash
HEAD_SUBJECT=$(git log -1 --format=%s)
case "$HEAD_SUBJECT" in
  "tinyc-mre: "*) ;;  # OK
  *)
    echo "ERROR: HEAD is not a tinyc-mre commit (subject: $HEAD_SUBJECT)."
    echo "Run the create-tinyc-mre skill first to produce an MRE."
    exit 1
    ;;
esac
```

If you find HEAD is not a `tinyc-mre:` commit, surface this to the user
verbatim and stop. Do **not** silently fall back to `git log --grep`,
`git stash`, or any other heuristic.

## Procedure

### Phase 0: Read Project Guidelines

Read `AGENTS.md` and `CLAUDE.md` (if present) at the repo root. Also read
`corec/AGENTS.md` if present. Follow them, but this SKILL.md takes
precedence.

### Phase 1: Read the MRE Commit

```bash
git log -1                     # full message — read Bug/Symptom/Hypothesis
git show --stat HEAD           # confirm only:
                               #   examples/tinyc/tests/<name>.tc
                               #   examples/tinyc/tests.toml
```

Extract:
- `<name>` from the subject (`tinyc-mre: <name>`).
- The **failure signature** from the body (this is your reproduction oracle).
- The **symptom class** (parse-error / emit-error / lowering-error /
  link-error / wrong-output) — this drives Phase 3.
- The **hypothesis**, if any.

### Phase 2: Reproduce the Bug (Single Test, Not Full Suite)

The MRE is one test file. Reproduce in seconds, not minutes:

```bash
NAME=<name>

# Reference: clang must succeed and emit the expected_stdout.
clang -O0 -x c -o /tmp/ref examples/tinyc/tests/$NAME.tc && /tmp/ref

# tinyc reproduction:
./tinyc --emit=llvm -o /tmp/$NAME.ll examples/tinyc/tests/$NAME.tc
# If that succeeds, link and run:
clang -nostdlib -lSystem /tmp/$NAME.ll -o /tmp/$NAME && /tmp/$NAME
```

Confirm the observed failure matches the **failure signature** from the
commit message. If it doesn't reproduce, stop and ask the user — the
environment is wrong, the submodule moved, or the bug is already fixed.

### Phase 3: Diagnose

The tinyC source layout (in `corec/examples/tinyc/`):

| Symptom | First place to look |
|---|---|
| `tinyc parse error ...` | `corec/examples/tinyc/parse.c` |
| Tokenizer / unknown token | `corec/examples/tinyc/lex.c` |
| `tinyc emit error ...` | `corec/examples/tinyc/emit.c` |
| Wrong MLIR / unknown op in output | `corec/examples/tinyc/emit.c` |
| MLIR lowering / pass-pipeline failure | this repo's `mlir_*.c`, `op_parsers.c` |
| Link error (undefined symbol) | missing extern decl or generated single-TU support |
| Wrong runtime output | `emit.c` codegen or corec-stdlib formatting |

Search for symbols / error strings:

```bash
grep -rn "<error text>" corec/examples/tinyc/
grep -rn "<construct keyword, e.g. _Generic>" corec/examples/tinyc/
```

Read the surrounding code to understand intended behavior before changing
anything.

### Phase 4: Implement the Fix

Edit files under `corec/examples/tinyc/` (the submodule). Keep changes
minimal and focused on the bug.

Rebuild and re-run the MRE (fast loop):

```bash
pixi run -e upstream build_tinyc_upstream
./tinyc --emit=llvm -o /tmp/$NAME.ll examples/tinyc/tests/$NAME.tc
clang -nostdlib -lSystem /tmp/$NAME.ll -o /tmp/$NAME && /tmp/$NAME
```

Iterate until the MRE passes (output matches the `expected_stdout` from
`tests.toml`).

### Phase 5: Run the Full tinyC Test Suite

```bash
pixi run -e upstream test_tinyc_upstream
```

All tinyC tests must pass. If any **other** test now fails:
- If your fix caused a regression, fix it.
- If the failure is pre-existing and unrelated, surface it to the user;
  do not paper over it.

### Phase 6: Run the Bootstrap Suite (Smoke Test)

Confirm the bootstrap pipeline still works:

```bash
pixi run -e macos test_macos_tinyc
```

If this used to pass and now fails, the fix has a regression. Iterate.
If it was already failing (some bugs aren't fixed yet), note this
explicitly to the user and move on.

### Phase 7: Commit the Fix

The fix lives in the `corec` submodule. There are two commits to make:

**(a) Inside `corec/`** — the actual source change:

```bash
cd corec
git checkout -b tinyc-fix-<name> 2>/dev/null || git switch tinyc-fix-<name>
git add -A
git commit -m "tinyc: fix <one-line summary>

<paragraph: what the bug was and what the fix does>

Regression test: examples/tinyc/tests/<name>.tc in the mlir repo
(MRE commit <short SHA>)."
cd ..
```

**(b) In the outer `mlir` repo** — bump the submodule pointer:

```bash
git add corec
git commit -m "Bump corec for tinyc fix: <one-line summary>

Fixes the bug captured by tinyc-mre commit <short SHA>:
examples/tinyc/tests/<name>.tc now passes."
```

Do **not** amend the original `tinyc-mre:` commit — keep it as a permanent
record of the bug. The fix sits on top as follow-up commits.

Do NOT include `Co-authored-by` trailers in any of these commits.

### Phase 8: Summary

Print:

```
tinyC bug fixed.

MRE:        <short SHA> tinyc-mre: <name>
Fix:        corec @ <short SHA> tinyc: fix <one-line>
Submodule:  mlir  @ <short SHA> Bump corec for tinyc fix: ...

Test status:
  examples/tinyc/tests/<name>.tc — PASS
  test_tinyc_upstream            — All N tinyC tests passed
  test_macos_tinyc               — <pass/fail count, with note>

Next: push the corec branch and open a PR; then push mlir and open a PR
that references the corec PR.
```

## Tips

- **Don't run the full suite to "find" the MRE.** The MRE is HEAD. Run
  only the one test until you're confident in the fix.
- **`emit.c` is most often the culprit.** Many tinyC bugs are codegen
  oversights for a specific construct combination (struct + fnptr,
  union + offset, va_arg + small types).
- **Read the hypothesis field.** `create-tinyc-mre` may already point at
  the right area.
- **Multi-bug MREs**: if one fix unblocks a different failure, that's a
  *new* MRE. Run `create-tinyc-mre` again, don't bundle.
- **Submodule discipline**: always commit inside `corec/` first, then
  bump the pointer in `mlir`. Never leave dirty submodule edits.
- **Iteration is fine**: if the first fix doesn't work or breaks other
  tests, amend the corec-side commit and re-bump. Just never amend the
  `tinyc-mre:` commit.

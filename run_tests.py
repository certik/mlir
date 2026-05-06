#!/usr/bin/env python
"""Test runner for the MLIR repo.

Each entry in tests/tests.toml is a single input file. References are
stored under tests/reference/<stem>.canonical.out, <stem>.classic.out,
and <stem>.generic.out. References are generated using the upstream
backend (treated as ground truth):

    canonical.out  =  ./parser_upstream FILE --parse=upstream --print=upstream
    classic.out    =  ./parser_upstream FILE --parse=upstream --print=classic
    generic.out    =  ./parser_upstream FILE --parse=upstream --print=generic

When a file is marked `upstream_parser = false` (e.g. it uses the
Triton dialect which is not in upstream MLIR), references fall back to
using the classic parser, and the canonical reference is omitted.

The runner then exercises every supported (parser, printer, backend)
combination and diffs against the matching reference.

Usage:
    python run_tests.py            # run all tests against committed refs
    python run_tests.py -u         # regenerate references
    python run_tests.py --upstream # only test the upstream backend
    python run_tests.py --native   # only test the native backend
"""
import argparse
import os
import subprocess
import sys
import shutil
import toml

ROOT = os.path.abspath(os.path.dirname(__file__))
TESTS = os.path.join(ROOT, "tests")
REF = os.path.join(TESTS, "reference")
OUT = os.path.join(TESTS, "output")

NATIVE = os.path.join(ROOT, "parser")
UPSTREAM = os.path.join(ROOT, "parser_upstream")
if os.name == "nt":
    NATIVE += ".exe"
    UPSTREAM += ".exe"


class Fail(Exception):
    pass


def run_capture(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout, p.stderr


def diff(a_path, b_path):
    """Return None if files match, or a diff string if they differ."""
    with open(a_path, "rb") as f:
        a = f.read()
    with open(b_path, "rb") as f:
        b = f.read()
    if a == b:
        return None
    # Use system diff for nice output
    try:
        return subprocess.run(
            ["diff", "-u", a_path, b_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ).stdout.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return f"<files differ: {a_path} vs {b_path}>"


def run_combo(exe, infile, parse_kind, print_kind):
    """Run `exe infile --parse=KIND --print=KIND`. Returns stdout bytes
    on success or raises Fail on non-zero exit."""
    cmd = [exe, infile, f"--parse={parse_kind}", f"--print={print_kind}"]
    rc, out, err = run_capture(cmd)
    if rc != 0:
        raise Fail(f"command failed (rc={rc}): {' '.join(cmd)}\nstderr:\n{err.decode('utf-8', errors='replace')}")
    return out


def stem(filename):
    return os.path.splitext(os.path.basename(filename))[0] + os.path.splitext(filename)[1].replace('.', '_')


def ref_path(filename, parse_kind, print_kind):
    # e.g. tests/reference/simple_mlir.upstream.upstream.out
    return os.path.join(REF, f"{stem(filename)}.{parse_kind}.{print_kind}.out")


# All (parse, print) combos we test. Each row is exercised on both backends
# (subject to backend support). When `upstream_parser` is false in tests.toml,
# only the classic-parser rows are exercised.
COMBOS_CLASSIC_PARSER = [
    ("classic", "classic"),
    ("classic", "generic"),
]
COMBOS_UPSTREAM_PARSER = [
    ("upstream", "upstream"),  # canonical
    ("upstream", "classic"),
    ("upstream", "generic"),
]


# Reference files known to fail upstream round-trip due to structural classic-
# printer bugs we have not yet fixed. Each entry should have a one-line note
# pointing at the *root cause* (not just the upstream error message). This
# list is enforced to only ever shrink: any ref that starts round-tripping
# must be removed (run_tests.py --validate-refs will fail until you do).
#
# Tracking these centrally — instead of letting them silently slip past CI —
# means new regressions are caught immediately while pre-existing structural
# bugs can be tackled one PR at a time.
VALIDATE_REFS_SKIP = {
    # Native classic parser doesn't fully parse `gpu.launch blocks(...) in
    # (...) threads(...) in (...)` (operands and outer-block sizes are lost),
    # so the regenerated ref carries a `gpu.launch` with 0 operands which
    # then fails to re-parse through upstream. Needs a native-parser fix.
    "t3_mlir.classic.classic.out",
    # Native classic parser is incomplete for several ops in d.mlir
    # (linalg.fill `ins/outs`, linalg.copy `outs`, tensor.extract result
    # type), producing invalid IR after round-trip. Needs native-parser
    # fixes for those ops.
    "d_mlir.classic.classic.out",
}


def ensure_refs(filename, upstream_parser):
    """Generate per-(parser,printer) reference files using parser_upstream."""
    infile = os.path.join(TESTS, filename)
    rows = list(COMBOS_CLASSIC_PARSER)
    if upstream_parser:
        rows.extend(COMBOS_UPSTREAM_PARSER)
    for parse_k, print_k in rows:
        try:
            out = run_combo(UPSTREAM, infile, parse_k, print_k)
        except Fail as e:
            raise Fail(f"failed to generate ref {parse_k}/{print_k} for {filename}: {e}")
        path = ref_path(filename, parse_k, print_k)
        with open(path, "wb") as f:
            f.write(out)


def check_combo(exe, filename, parse_kind, print_kind):
    """Run a (parse, print, backend) combo and diff against the matching
    {parse}.{print} reference file."""
    infile = os.path.join(TESTS, filename)
    ref = ref_path(filename, parse_kind, print_kind)
    if not os.path.exists(ref):
        raise Fail(f"missing reference {ref}")
    out = run_combo(exe, infile, parse_kind, print_kind)
    out_path = os.path.join(OUT, f"{stem(filename)}.{os.path.basename(exe)}.{parse_kind}.{print_kind}.out")
    with open(out_path, "wb") as f:
        f.write(out)
    d = diff(ref, out_path)
    if d is not None:
        raise Fail(
            f"{os.path.basename(exe)} {filename} --parse={parse_kind} --print={print_kind}"
            f" differs from reference {os.path.basename(ref)}:\n{d}"
        )


def discover_test_files():
    """List all *.mlir and *.ttir files directly in tests/."""
    out = []
    for name in sorted(os.listdir(TESTS)):
        if name.endswith(".mlir") or name.endswith(".ttir"):
            out.append(name)
    return out


def assert_toml_covers_all_files(tests):
    """Every test file on disk must appear in tests.toml. This catches files
    that are added but forgotten in the registry (which would silently never
    run)."""
    on_disk = set(discover_test_files())
    in_toml = {t["filename"] for t in tests}
    missing = on_disk - in_toml
    extra = in_toml - on_disk
    msgs = []
    if missing:
        msgs.append(f"tests/ has files not listed in tests.toml: {sorted(missing)}")
    if extra:
        msgs.append(f"tests.toml lists files not on disk: {sorted(extra)}")
    if msgs:
        for m in msgs:
            print(f"error: {m}", file=sys.stderr)
        sys.exit(1)


def assert_combos_cover_kinds():
    """Sanity-check that the COMBOS arrays exercise every kind exposed by
    the public API. If a new kind is added to MLIR_PrintKind/MLIR_ParseKind,
    this guard forces us to update the test matrix."""
    known_print = {"upstream", "classic", "generic"}
    known_parse = {"upstream", "classic"}
    seen_print = set()
    seen_parse = set()
    for parse_k, print_k in COMBOS_CLASSIC_PARSER + COMBOS_UPSTREAM_PARSER:
        seen_parse.add(parse_k)
        seen_print.add(print_k)
    if seen_print != known_print or seen_parse != known_parse:
        print(
            f"error: COMBOS in run_tests.py do not cover all API kinds.\n"
            f"  parse: have {sorted(seen_parse)}, expect {sorted(known_parse)}\n"
            f"  print: have {sorted(seen_print)}, expect {sorted(known_print)}",
            file=sys.stderr,
        )
        sys.exit(1)


def roundtrip_validate_refs(tests):
    """Validate that every classic-form reference is syntactically valid MLIR
    by feeding it back through `parser_upstream --parse=upstream`. This proves
    our classic printer emits MLIR upstream accepts (not just text that our
    own parser likes).

    We skip *.generic.out because our generic format is a debugging
    representation, not MLIR's standard generic form. We skip
    *.upstream.upstream.out because it is by definition upstream's own
    pretty form (already valid).

    Refs in VALIDATE_REFS_SKIP are known-broken due to structural classic-
    printer bugs (affine.for syntax, memref subscript syntax, attribute "..."
    placeholders, SSA renumbering, etc.). They are skipped here but the list
    is enforced to only ever shrink: if a skipped ref now passes round-trip,
    we fail and ask for it to be removed from the list."""
    if not os.path.exists(UPSTREAM):
        print(f"error: {UPSTREAM} required for round-trip ref validation", file=sys.stderr)
        sys.exit(1)
    failures = []
    unexpected_passes = []
    n = 0
    for t in tests:
        filename = t["filename"]
        upstream_parser = t.get("upstream_parser", True)
        if not upstream_parser:
            # Files using dialects unknown to upstream MLIR (e.g. Triton)
            # cannot be re-parsed by upstream regardless of how they were
            # printed; skip.
            continue
        for parse_k, print_k in COMBOS_CLASSIC_PARSER + COMBOS_UPSTREAM_PARSER:
            if print_k != "classic":
                continue
            ref = ref_path(filename, parse_k, print_k)
            if not os.path.exists(ref):
                continue
            base = os.path.basename(ref)
            cmd = [UPSTREAM, ref, "--parse=upstream", "--print=upstream"]
            rc, _, err = run_capture(cmd)
            if base in VALIDATE_REFS_SKIP:
                if rc == 0:
                    unexpected_passes.append(base)
                continue
            n += 1
            if rc != 0:
                failures.append((ref, err.decode("utf-8", errors="replace")))
    if unexpected_passes:
        print(
            f"\n{len(unexpected_passes)} ref(s) listed in VALIDATE_REFS_SKIP now "
            f"round-trip cleanly. Please remove them from the list in run_tests.py:",
            file=sys.stderr,
        )
        for b in unexpected_passes:
            print(f"  {b}", file=sys.stderr)
        sys.exit(1)
    if failures:
        print(f"\n{len(failures)}/{n} reference file(s) failed upstream round-trip:\n", file=sys.stderr)
        for ref, err in failures:
            print(f"--- {os.path.basename(ref)} ---", file=sys.stderr)
            print(err, file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--update", action="store_true",
                    help="regenerate references")
    ap.add_argument("-s", "--sequential", action="store_true",
                    help="(accepted for compatibility; tests already run sequentially)")
    ap.add_argument("--upstream", action="store_true",
                    help="run upstream-backend tests only")
    ap.add_argument("--native", action="store_true",
                    help="run native-backend tests only")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--validate-refs", action="store_true",
                    help="validate that every *.classic.out reference parses cleanly through "
                         "`parser_upstream --parse=upstream`. Surfaces classic-printer output "
                         "that upstream MLIR rejects. Currently surfaces pre-existing bugs "
                         "for affine.for, memref.alloc operand-list, and unnamed func args; "
                         "fix in follow-up PRs. Not run in normal CI.")
    args = ap.parse_args()

    do_native = (not args.upstream) or args.native
    do_upstream = (not args.native) or args.upstream

    os.makedirs(REF, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    cfg = toml.load(os.path.join(TESTS, "tests.toml"))
    tests = cfg["test"]

    # Static guardrails (run before any test execution).
    assert_combos_cover_kinds()
    assert_toml_covers_all_files(tests)
    if not tests:
        print("error: tests.toml has no [[test]] entries", file=sys.stderr)
        sys.exit(1)

    if args.update:
        # Regenerate references; require parser_upstream.
        if not os.path.exists(UPSTREAM):
            print(f"error: {UPSTREAM} is required to regenerate references", file=sys.stderr)
            sys.exit(1)
        # Wipe stale refs.
        if os.path.isdir(REF):
            shutil.rmtree(REF)
        os.makedirs(REF, exist_ok=True)
        for t in tests:
            filename = t["filename"]
            upstream_parser = t.get("upstream_parser", True)
            print(f"  ref {filename} (upstream_parser={upstream_parser})")
            ensure_refs(filename, upstream_parser)
        print("References updated.")
        return

    if args.validate_refs:
        roundtrip_validate_refs(tests)
        print("All classic references round-trip through upstream parser.")
        return

    # Hard-fail on missing binaries instead of silently skipping the
    # corresponding combos. The runner is meaningless without both.
    if do_native and not os.path.exists(NATIVE):
        print(f"error: {NATIVE} not found (run `pixi r build_<platform>` first)", file=sys.stderr)
        sys.exit(1)
    if do_upstream and not os.path.exists(UPSTREAM):
        print(f"error: {UPSTREAM} not found (run `pixi r build_parser_upstream` first)", file=sys.stderr)
        sys.exit(1)

    failures = []
    n = 0
    for t in tests:
        filename = t["filename"]
        upstream_parser = t.get("upstream_parser", True)
        # Native backend: classic-parser combos only.
        native_combos = list(COMBOS_CLASSIC_PARSER)
        # Upstream backend: classic-parser combos always; upstream-parser
        # combos only if the upstream parser handles the file.
        upstream_combos = list(COMBOS_CLASSIC_PARSER)
        if upstream_parser:
            upstream_combos.extend(COMBOS_UPSTREAM_PARSER)

        if do_native:
            for parse_k, print_k in native_combos:
                n += 1
                desc = f"native {filename} parse={parse_k} print={print_k}"
                if args.verbose:
                    print(f"  {desc}")
                try:
                    check_combo(NATIVE, filename, parse_k, print_k)
                except Fail as e:
                    failures.append((desc, str(e)))
        if do_upstream:
            for parse_k, print_k in upstream_combos:
                n += 1
                desc = f"upstream {filename} parse={parse_k} print={print_k}"
                if args.verbose:
                    print(f"  {desc}")
                try:
                    check_combo(UPSTREAM, filename, parse_k, print_k)
                except Fail as e:
                    failures.append((desc, str(e)))

    if n == 0:
        print("error: zero test combos ran (refusing to report success)", file=sys.stderr)
        sys.exit(1)

    if failures:
        print(f"\n{len(failures)}/{n} FAILED:\n")
        for desc, err in failures:
            print(f"--- {desc} ---")
            print(err)
            print()
        sys.exit(1)
    print(f"All {n} tests passed.")


if __name__ == "__main__":
    main()

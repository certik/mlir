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
TINYC = Path(os.environ.get("TINYC", str(ROOT / ("tinyc.exe" if IS_WIN else "tinyc")))).resolve()
RUNTIME = HERE / "runtime.c"
RUNTIME_WASM = HERE / "runtime_wasm.c"
RUNTIME_WASM_START = HERE / "start_wasm.s"
# corec platform_<os>.c whose file-I/O + exit primitives the via-wasm Mach-O
# backend's WASI adapters call (spliced in via --host-platform).
HOST_PLATFORM = ROOT / "corec" / "platform" / "platform_macos.c"
# corec platform_linux.c compiled into the direct x86_64->ELF module so the
# runtime's I/O binds to the real platform layer (platform_fd_write) instead of
# a synthesized syscall thunk.
HOST_PLATFORM_ELF = ROOT / "corec" / "platform" / "platform_linux.c"
COREC_DIR = ROOT / "corec"
TESTS_TOML = HERE / "tests.toml"

CC = os.environ.get("CC", "cl" if IS_WIN else "clang")
LLC = os.environ.get("LLC", "llc")
WASM_LD = os.environ.get("WASM_LD", "wasm-ld")
WASMTIME = os.environ.get("WASMTIME", "wasmtime")
WASM_CC = os.environ.get("WASM_CC", "clang")

# When TINYC_USE_NATIVE_LINK=1, link wasm32 objects with the in-tree
# `mlir_wasm_link` linker instead of host wasm-ld. The native linker
# code lives in `mlir_wasm_link.c` and is compiled into both `tinyc`
# and `tinyc_native`, so by default we reuse whichever binary is
# already in $TINYC (no separate build needed). Set NATIVE_LINK
# explicitly to point at a different binary if needed.
USE_NATIVE_LINK = os.environ.get("TINYC_USE_NATIVE_LINK") == "1"
NATIVE_LINK = Path(os.environ.get("NATIVE_LINK", str(TINYC))).resolve()

# Backend used by `tinyc --lowering=...`. If unset, no `--lowering=` flag
# is passed and each binary uses its own default (upstream for tinyc,
# native for tinyc_native). Set TINYC_LOWERING=upstream or =native to
# force a specific path through the suite.
LOWERING = os.environ.get("TINYC_LOWERING")
LOWERING_FLAG = [f"--lowering={LOWERING}"] if LOWERING else []

# Code-generation/runtime target for the suite. "native" (default) emits
# LLVM IR via tinyc, then llc + host CC + runtime.c. "wasm" emits a
# wasm32 object via tinyc, then wasm-ld + runtime_wasm.c, and runs the
# resulting .wasm via wasmtime. Both TINYC_LOWERING values are valid
# with the wasm target.
TARGET = os.environ.get("TINYC_TARGET", "native")
# Selects which macho backend to use when TARGET=macho. The default
# `wasm` backend goes wasmssa -> wasmstack -> wasm.o -> Mach-O. The
# `llvm` backend goes wasmssa -> llvm -> aarch64 -> Mach-O (native,
# LP64); `llvm_via_wasm` lifts a linked wasm module through it.
MACHO_BACKEND = os.environ.get("TINYC_MACHO_BACKEND", "wasm")
# When TINYC_LIFT_USE_NATIVE=1, the upstream tinyc binary first runs the
# (partial) native cf->scf lifter and then finishes any leftover cf ops
# with upstream's CFGToSCF pass. That fallback handles all the patterns
# the native lifter doesn't yet cover (loops with break/continue/early-
# return, switch cascades, etc), so wasm_native_skip should NOT skip in
# this configuration.
LIFT_USE_NATIVE = os.environ.get("TINYC_LIFT_USE_NATIVE") == "1"


def run(cmd, **kw):
    kw.setdefault("timeout", 60)
    try:
        return subprocess.run(cmd, capture_output=True, text=True, **kw)
    except subprocess.TimeoutExpired as e:
        class _R:
            pass
        r = _R()
        r.returncode = -124
        r.stdout = e.stdout or ""
        r.stderr = (e.stderr or "") + "\n[timeout]"
        return r


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


def build_wasm_runtime(out_path: Path, start_obj: Path):
    """Compile runtime_wasm.c (and the _start asm shim) into wasm32
    relocatable object files once per test run. Cached on disk; only
    rebuilt if missing or older than the source."""
    runtime_stale = (
        not out_path.exists()
        or out_path.stat().st_mtime < RUNTIME_WASM.stat().st_mtime
    )
    if runtime_stale:
        r = run([
            WASM_CC, "--target=wasm32-wasi",
            "-nostdlib", "-nostdlibinc", "-fno-builtin", "-O1",
            "-c", str(RUNTIME_WASM), "-o", str(out_path),
        ])
        if r.returncode != 0:
            return r
    start_stale = (
        not start_obj.exists()
        or start_obj.stat().st_mtime < RUNTIME_WASM_START.stat().st_mtime
    )
    if start_stale:
        r = run([
            WASM_CC, "--target=wasm32",
            "-c", str(RUNTIME_WASM_START), "-o", str(start_obj),
        ])
        if r.returncode != 0:
            return r
    return None


def link_wasm(obj_paths, runtime_obj: Path, start_obj: Path,
              wasm_path: Path):
    """Link tinyc-emitted wasm32 object(s) together with the wasm runtime
    + _start shim. Accepts either a single Path or a list of Paths so
    that multi-object tests (one .wasm.o per source) can be linked in
    the same call. Either uses host `wasm-ld` (default) or the in-tree
    native linker (`TINYC_USE_NATIVE_LINK=1`)."""
    if isinstance(obj_paths, Path):
        obj_paths = [obj_paths]
    obj_args = [str(p) for p in obj_paths]
    if USE_NATIVE_LINK:
        return run([
            str(NATIVE_LINK), "--link",
            "-o", str(wasm_path),
            "--export=_start",
            *obj_args, str(runtime_obj), str(start_obj),
        ])
    return run([
        WASM_LD, "--no-entry", "--export=_start",
        *obj_args, str(runtime_obj), str(start_obj),
        "-o", str(wasm_path),
    ])


def main():
    if not TINYC.exists():
        print(f"error: {TINYC} not found; run build_tinyc_upstream first", file=sys.stderr)
        return 2
    cfg = tomllib.loads(TESTS_TOML.read_text())
    tests = cfg.get("test", [])
    if not tests:
        print("error: no tests found in tests.toml", file=sys.stderr)
        return 2

    # Platform key matching `platforms = [...]` in tests.toml. A test
    # with a `platforms` field runs only on listed platforms. Values:
    # "linux", "darwin", "win32".
    plat = sys.platform if sys.platform.startswith(("linux", "darwin", "win32")) else sys.platform
    if plat.startswith("linux"):
        plat_key = "linux"
    elif plat.startswith("darwin"):
        plat_key = "darwin"
    elif plat.startswith("win32"):
        plat_key = "win32"
    else:
        plat_key = plat

    # The native lowering is allowed with the wasm target: the upstream
    # `MLIR_TranslateModuleToWasm` is responsible for the LLVM->wasm step,
    # and the lowering choice only affects how MLIR is reduced to the LLVM
    # dialect beforehand.

    # Pre-build the wasm runtime object once per run.
    wasm_runtime_obj = HERE / "tests" / "runtime_wasm.o"
    wasm_start_obj   = HERE / "tests" / "start_wasm.o"
    if TARGET == "wasm" or TARGET == "macho":
        r = build_wasm_runtime(wasm_runtime_obj, wasm_start_obj)
        if r is not None and r.returncode != 0:
            print(f"error: failed to compile wasm runtime\nstderr:\n{r.stderr}",
                  file=sys.stderr)
            return 2

    # The macho backend only runs on Apple Silicon (arm64). The binary it
    # produces is a Mach-O ARM64 executable.
    if TARGET == "macho":
        import platform as _platform
        if sys.platform != "darwin" or _platform.machine() != "arm64":
            print(f"error: TINYC_TARGET=macho is only supported on Darwin/arm64 "
                  f"(got sys.platform={sys.platform!r}, machine={_platform.machine()!r})",
                  file=sys.stderr)
            return 2

    # The elf backend produces a static x86_64 Linux ELF executable (direct
    # syscalls, no libc), so it only runs on Linux/x86_64.
    if TARGET == "elf":
        import platform as _platform
        if not sys.platform.startswith("linux") or _platform.machine() not in ("x86_64", "amd64"):
            print(f"error: TINYC_TARGET=elf is only supported on Linux/x86_64 "
                  f"(got sys.platform={sys.platform!r}, machine={_platform.machine()!r})",
                  file=sys.stderr)
            return 2

    failures = 0
    skipped = 0
    for t in tests:
        name = t["name"]
        # `expected_stdout` is optional (default ""): macho-only tests that
        # don't exercise stdout don't need to set it. `expected_exit_code`
        # is also optional (default 0): macho_exit-style tests set it.
        expected         = t.get("expected_stdout", "")
        expected_rc      = t.get("expected_exit_code", 0)
        # Tests that differ between wasm32 (4-byte pointers, 4-byte long)
        # and the host (8-byte pointers on x86_64/aarch64) can override
        # the expected stdout with `expected_stdout_wasm`. The wasm and
        # llvm_via_wasm macho backends consume the wasm32-sized pipeline,
        # so they use the wasm32-flavored expected stdout. The native
        # `llvm` macho backend
        # is LP64 (native sizing), so it keeps the native `expected_stdout`.
        if (TARGET in ("wasm", "macho") and "expected_stdout_wasm" in t
                and not (TARGET == "macho" and MACHO_BACKEND == "llvm")):
            expected = t["expected_stdout_wasm"]
        # `targets` lists the runner targets a test may run under. Default
        # (omitted) is ["native", "wasm"] — the established backends.
        # Mach-O-only tests opt in explicitly with `targets = ["macho"]`.
        targets = t.get("targets", ["native", "wasm"])
        if TARGET not in targets:
            print(f"SKIP {name} (targets={targets}, current={TARGET})")
            skipped += 1
            continue
        platforms = t.get("platforms")
        if platforms is not None and plat_key not in platforms:
            print(f"SKIP {name} (platforms={platforms}, current={plat_key})")
            skipped += 1
            continue
        # Optional opt-out for the wasm target only (e.g. tests that
        # rely on host-libc behavior wasm32-wasi can't reproduce).
        # The macho backend consumes the same wasm pipeline output, so
        # `wasm_skip` implies a macho skip too.
        if TARGET in ("wasm", "macho") and t.get("wasm_skip"):
            print(f"SKIP {name} (wasm_skip)")
            skipped += 1
            continue
        # Inverse opt-out: tests that exercise wasm-only builtins
        # (e.g. `__builtin_wasm_memory_size`) cannot be run on the
        # native (LLVM-IR / llc) target — those intrinsics aren't
        # valid for x86/aarch64 codegen.
        if TARGET == "native" and t.get("native_skip"):
            print(f"SKIP {name} (native_skip)")
            skipped += 1
            continue
        # Optional opt-out for the native LLVM->WASM pipeline only
        # (mlir_llvm_to_wasmssa.c + mlir_wasmssa_to_wasmstack.c +
        # mlir_wasmstack_to_bin.c). Covers the few corners the in-tree
        # pipeline doesn't implement yet; the upstream wasm path still
        # runs the test.
        if (TARGET == "wasm" and LOWERING == "native"
                and not LIFT_USE_NATIVE
                and t.get("wasm_native_skip")):
            print(f"SKIP {name} (wasm_native_skip)")
            skipped += 1
            continue
        # Inverse opt-out: tests that exercise wasm-specific custom
        # function attributes (e.g.
        # `__attribute__((__import_module__("...")))`) which are
        # honored by the in-tree native LLVM->WASM pipeline but not
        # by the upstream LLVM WebAssembly backend (which would need
        # a separate `passthrough = ...` attribute plumbing to forward
        # them through MLIR's `convert-func-to-llvm` pass).
        if (TARGET == "wasm" and LOWERING != "native"
                and t.get("wasm_upstream_skip")):
            print(f"SKIP {name} (wasm_upstream_skip)")
            skipped += 1
            continue
        # Inverse opt-out: tests that exercise wasm-only semantics (WASI
        # imports, wasm32 long-sizing) which cannot run on the LP64
        # `llvm` macho backend — there is no WASI host there.
        if (TARGET == "macho" and MACHO_BACKEND == "llvm"
                and t.get("macho_llvm_skip")):
            print(f"SKIP {name} (macho_llvm_skip)")
            skipped += 1
            continue
        # Inverse opt-in: tests that only make sense on the direct `llvm` macho
        # backend (not the wasm / llvm_via_wasm macho backends, which compile
        # through the wasm pipeline first). Currently unused, but kept as a
        # general gate for native-aarch64-only behaviour.
        if t.get("macho_llvm_only") and not (TARGET == "macho" and MACHO_BACKEND == "llvm"):
            print(f"SKIP {name} (macho_llvm_only)")
            skipped += 1
            continue
        # Multi-file tests pass `sources = [...]`; single-file tests
        # default to `<name>.tc` for backwards compatibility.
        sources = t.get("sources", [f"{name}.tc"])
        srcs = [HERE / "tests" / s for s in sources]

        if TARGET == "wasm":
            obj  = HERE / "tests" / f"{name}.wasm.o"
            wasm = HERE / "tests" / f"{name}.wasm"

            # When `link_separately = true`, compile each source into
            # its own .wasm.o and link them with wasm-ld (or the
            # in-tree linker). This exercises cross-object symbol
            # resolution — the default flow of "merge all sources into
            # one MLIR module first" cannot regress static-linkage
            # bugs that only manifest across object boundaries.
            link_separately = bool(t.get("link_separately"))
            objs = []

            if link_separately:
                fail = False
                for src in srcs:
                    obj_i = HERE / "tests" / f"{src.stem}.wasm.o"
                    r = run([str(TINYC), "--emit=wasm", *LOWERING_FLAG,
                             "-I", str(HERE / "tests"),
                             "-o", str(obj_i),
                             str(src)])
                    if r.returncode != 0:
                        print(f"FAIL {name}: tinyc returned {r.returncode} on {src.name}\nstderr:\n{r.stderr}")
                        failures += 1
                        fail = True
                        break
                    objs.append(obj_i)
                if fail:
                    continue
            else:
                # Stage 1: tinyc emits wasm32 object directly.
                r = run([str(TINYC), "--emit=wasm", *LOWERING_FLAG,
                         "-I", str(HERE / "tests"),
                         "-o", str(obj),
                         *[str(s) for s in srcs]])
                if r.returncode != 0:
                    print(f"FAIL {name}: tinyc returned {r.returncode}\nstderr:\n{r.stderr}")
                    failures += 1
                    continue
                objs = [obj]

            # Stage 2: link with wasm-ld + runtime_wasm.o + start_wasm.o.
            r = link_wasm(objs, wasm_runtime_obj, wasm_start_obj, wasm)
            if r.returncode != 0:
                print(f"FAIL {name}: wasm-ld failed\nstderr:\n{r.stderr}\nstdout:\n{r.stdout}")
                failures += 1
                continue

            # Stage 3: run under wasmtime.
            r = run([WASMTIME, str(wasm)])
            if r.returncode != expected_rc:
                print(f"FAIL {name}: wasm exited with status {r.returncode} (expected {expected_rc})\nstdout: {r.stdout!r}\nstderr: {r.stderr!r}")
                failures += 1
                continue
            if r.stdout != expected:
                print(f"FAIL {name}: stdout mismatch\n  expected: {expected!r}\n  got:      {r.stdout!r}")
                failures += 1
                continue

            print(f"PASS {name}")
            continue

        if TARGET == "elf":
            # Native x86_64 -> static ELF executable (Linux). One invocation
            # compiles + links + emits a directly-runnable binary; run it and
            # compare stdout + exit code against the LP64 expectations.
            exe = HERE / "tests" / f"{name}.elf"
            tinyc_cmd = [str(TINYC), "--emit=elf", *LOWERING_FLAG,
                         "--host-platform", str(HOST_PLATFORM_ELF),
                         "-I", str(COREC_DIR), "-I", str(HERE / "tests"),
                         "-o", str(exe)]
            tinyc_cmd.extend(str(s) for s in srcs)
            r = run(tinyc_cmd)
            if r.returncode != 0:
                print(f"FAIL {name}: tinyc returned {r.returncode}\nstderr:\n{r.stderr}")
                failures += 1
                continue
            r = run([str(exe)])
            if r.returncode != expected_rc:
                print(f"FAIL {name}: elf exited with status {r.returncode} (expected {expected_rc})\nstdout: {r.stdout!r}\nstderr: {r.stderr!r}")
                failures += 1
                continue
            if r.stdout != expected:
                print(f"FAIL {name}: stdout mismatch\n  expected: {expected!r}\n  got:      {r.stdout!r}")
                failures += 1
                continue
            print(f"PASS {name}")
            continue

        if TARGET == "macho":
            exe = HERE / "tests" / f"{name}.macho"

            # The llvm_via_wasm sub-backend goes through the lifter:
            #   per-source --emit=wasm + wasm-ld + --from-wasm <linked.wasm>
            # so it always exercises the full wasm-linker pipeline before
            # touching the llvm backend. This is the only way to get
            # per-TU isolation that tinyc itself doesn't provide for the
            # joint multi-source --emit=macho invocation.
            if MACHO_BACKEND == "llvm_via_wasm":
                via_backend = "llvm"
                # Per-source emit -> link -> from-wasm -> macho.
                # Mirror the wasm-target convention: by default the
                # sources are compiled jointly (so cross-file calls
                # without a forward decl resolve, etc); with
                # link_separately = true each source is compiled to
                # its own .wasm.o to exercise wasm-ld symbol
                # resolution. The latter is the path that the
                # existing test_tinyc_wasm task uses.
                link_separately = bool(t.get("link_separately"))
                fail = False
                wasm_objs = []
                if link_separately:
                    for src in srcs:
                        obj_i = HERE / "tests" / f"{src.stem}.macho.wasm.o"
                        r = run([str(TINYC), "--emit=wasm", *LOWERING_FLAG,
                                 "-I", str(HERE / "tests"),
                                 "-o", str(obj_i),
                                 str(src)])
                        if r.returncode != 0:
                            print(f"FAIL {name}: tinyc --emit=wasm on {src.name} returned {r.returncode}\nstderr:\n{r.stderr}")
                            failures += 1
                            fail = True
                            break
                        wasm_objs.append(obj_i)
                else:
                    obj_j = HERE / "tests" / f"{name}.macho.wasm.o"
                    r = run([str(TINYC), "--emit=wasm", *LOWERING_FLAG,
                             "-I", str(HERE / "tests"),
                             "-o", str(obj_j),
                             *[str(s) for s in srcs]])
                    if r.returncode != 0:
                        print(f"FAIL {name}: tinyc --emit=wasm returned {r.returncode}\nstderr:\n{r.stderr}")
                        failures += 1
                        fail = True
                    else:
                        wasm_objs.append(obj_j)
                if fail:
                    continue
                linked_wasm = HERE / "tests" / f"{name}.macho.linked.wasm"
                r = link_wasm(wasm_objs, wasm_runtime_obj, wasm_start_obj, linked_wasm)
                if r.returncode != 0:
                    print(f"FAIL {name}: wasm-ld failed\nstderr:\n{r.stderr}\nstdout:\n{r.stdout}")
                    failures += 1
                    continue
                r = run([str(TINYC), "--from-wasm", str(linked_wasm),
                         "--emit=macho", f"--macho-backend={via_backend}",
                         "--host-platform", str(HOST_PLATFORM),
                         "-I", str(COREC_DIR), "-I", str(ROOT),
                         "-o", str(exe)])
                if r.returncode != 0:
                    print(f"FAIL {name}: tinyc --from-wasm returned {r.returncode}\nstderr:\n{r.stderr}")
                    failures += 1
                    continue
                # Same retry-on-SIGKILL dance as the regular macho path.
                r = run([str(exe)])
                if r.returncode == -9:
                    r = run([str(exe)])
                if r.returncode != expected_rc:
                    print(f"FAIL {name}: macho exited with status {r.returncode} (expected {expected_rc})\nstdout: {r.stdout!r}\nstderr: {r.stderr!r}")
                    failures += 1
                    continue
                if r.stdout != expected:
                    print(f"FAIL {name}: stdout mismatch\n  expected: {expected!r}\n  got:      {r.stdout!r}")
                    failures += 1
                    continue
                print(f"PASS {name}")
                continue

            # Stage 1: tinyc emits wasm32 object, links it with the wasm
            # runtime + _start shim, and translates the linked module to
            # a signed Mach-O ARM64 binary — all in one invocation.
            # The `llvm` backend uses native host pointers and synthesises
            # its own runtime, so the wasm runtime objects are not needed.
            tinyc_cmd = [str(TINYC), "--emit=macho", *LOWERING_FLAG,
                         "-I", str(HERE / "tests"),
                         "-o", str(exe)]
            if MACHO_BACKEND == "llvm":
                tinyc_cmd.append("--macho-backend=llvm")
            else:
                tinyc_cmd.append(f"--wasm-runtime-obj={wasm_runtime_obj}")
                tinyc_cmd.append(f"--wasm-runtime-obj={wasm_start_obj}")
            tinyc_cmd.extend(str(s) for s in srcs)
            r = run(tinyc_cmd)
            if r.returncode != 0:
                print(f"FAIL {name}: tinyc returned {r.returncode}\nstderr:\n{r.stderr}")
                failures += 1
                continue

            # Stage 2: run the produced Mach-O binary directly. macOS
            # AMFI occasionally SIGKILLs freshly ad-hoc-codesigned
            # binaries the first time they're launched (the kernel's
            # signature-verification daemon races with the kernel
            # exec path). Retry once on SIGKILL — the second launch
            # always succeeds because the signature is now cached.
            r = run([str(exe)])
            if r.returncode == -9:
                r = run([str(exe)])
            if r.returncode != expected_rc:
                print(f"FAIL {name}: macho exited with status {r.returncode} (expected {expected_rc})\nstdout: {r.stdout!r}\nstderr: {r.stderr!r}")
                failures += 1
                continue
            if r.stdout != expected:
                print(f"FAIL {name}: stdout mismatch\n  expected: {expected!r}\n  got:      {r.stdout!r}")
                failures += 1
                continue

            print(f"PASS {name}")
            continue

        ll  = HERE / "tests" / f"{name}.ll"
        obj = HERE / "tests" / (f"{name}.obj" if IS_WIN else f"{name}.o")
        exe = HERE / "tests" / (f"{name}.exe" if IS_WIN else f"{name}.bin")

        # Stage 1: emit LLVM IR. The driver accepts multiple input files
        # and merges them into a single MLIR module.
        # `-I tests` so multi-file tests can `#include "shared.h"`.
        r = run([str(TINYC), "--emit=llvm", *LOWERING_FLAG,
                 "-I", str(HERE / "tests"),
                 *[str(s) for s in srcs]])
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
        if r.returncode != expected_rc:
            print(f"FAIL {name}: binary exited with status {r.returncode} (expected {expected_rc})\nstdout: {r.stdout!r}\nstderr: {r.stderr!r}")
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
    ran = len(tests) - skipped
    if skipped:
        print(f"\nAll {ran} tinyC tests passed ({skipped} skipped on {plat_key})")
    else:
        print(f"\nAll {len(tests)} tinyC tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""tinyC end-to-end test runner.

For each test in `tests.toml`, generate a Clang-compatible single-TU C
wrapper that includes corec + corec-stdlib + the test sources, then compile
that one wrapper with tinyC.
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
COREC_STDLIB_DIR = ROOT / "corec-stdlib"
COREC_STDLIB_COREC_DIR = COREC_STDLIB_DIR / "corec"
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
# LLVM IR via tinyc, then llc + host CC. "wasm" emits a
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


def _inc(path: Path) -> str:
    """Return a wrapper-local include path using portable separators."""
    return os.path.relpath(path, HERE / "tests").replace(os.sep, "/")


COREC_BASE_SOURCES = [
    "base/assert.c",
    "base/exit.c",
    "base/mem.c",
    "base/string.c",
    "base/numconv.c",
    "base/format.c",
    "base/strbuf.c",
    "base/io.c",
    "base/buddy.c",
    "base/arena.c",
    "base/scratch.c",
]

STDLIB_SOURCES = [
    "stdlib/string_impl.c",
    "stdlib/stdlib.c",
    "stdlib/stdio.c",
    "stdlib/printf.c",
]


def uses_wasm_runtime() -> bool:
    return TARGET == "wasm" or (TARGET == "macho" and MACHO_BACKEND != "llvm")


def use_unity_source() -> bool:
    return not (IS_WIN and TARGET == "native")


def uses_inline_platform() -> bool:
    return TARGET == "native" or (TARGET == "macho" and MACHO_BACKEND == "llvm")


def platform_source_for_unity() -> Path:
    if uses_wasm_runtime():
        return COREC_STDLIB_COREC_DIR / "platform" / "platform_wasm.c"
    if TARGET == "elf":
        return COREC_STDLIB_COREC_DIR / "platform" / "platform_linux.c"
    if IS_WIN:
        return COREC_STDLIB_COREC_DIR / "platform" / "platform_windows.c"
    if sys.platform.startswith("darwin"):
        return COREC_STDLIB_COREC_DIR / "platform" / "platform_macos.c"
    return COREC_STDLIB_COREC_DIR / "platform" / "platform_linux.c"


def unity_uses_host_main() -> bool:
    return TARGET != "elf"


def main_takes_args(srcs: list[Path]) -> bool:
    for src in srcs:
        text = src.read_text()
        if "main(int argc" in text or "main(int pargc" in text:
            return True
    return False


def write_unity_source(name: str, srcs: list[Path]) -> Path:
    """Generate a single Clang/tinyC-compatible TU for one test."""
    target_suffix = TARGET if TARGET != "macho" else f"macho_{MACHO_BACKEND}"
    unity = HERE / "tests" / f"{name}.{target_suffix}.single_tu.c"
    host_main = unity_uses_host_main()
    lines = [
        "/* Generated by run_tinyc_tests.py. Do not edit. */",
        "#define SINGLE_TU_BUILD 1",
        "#define COREC_STDLIB_PROVIDES_MEM 1",
        "#define PLATFORM_SKIP_ENTRY 1",
    ]
    lines += [
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#include <stdarg.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "",
    ]
    if host_main:
        lines.append("#define main app_main")
    for src in srcs:
        lines.append(f'#include "{_inc(src)}"')
    if host_main:
        lines.append("#undef main")
    lines += ["#undef _tinyc_print", ""]
    for src in COREC_BASE_SOURCES:
        lines.append(f'#include "{_inc(COREC_STDLIB_COREC_DIR / src)}"')
    stdlib_sources = list(STDLIB_SOURCES)
    if uses_wasm_runtime():
        # runtime_wasm.c supplies printf/vprintf for wasm-shaped tests. Keep the
        # rest of stdio (FILE, fputs, fputc, fopen, ...) in the unity root.
        stdlib_sources.remove("stdlib/printf.c")
    if TARGET == "native":
        # Native tests link against the host C library; use its variadic printf
        # ABI instead of compiling corec-stdlib's printf with tinyC.
        stdlib_sources.remove("stdlib/printf.c")
    for src in stdlib_sources:
        lines.append(f'#include "{_inc(COREC_STDLIB_DIR / src)}"')
    lines += ["", ""]
    if TARGET == "native":
        if sys.platform.startswith("linux"):
            lines += [
                "struct tinyc_va_list_x64 { unsigned int gp_offset; unsigned int fp_offset; void *overflow_arg_area; void *reg_save_area; };",
                "long long tinyc_va_arg_i64(struct tinyc_va_list_x64 *ap) {",
                "  char *p;",
                "  if (ap->gp_offset < 48) { p = (char*)ap->reg_save_area + ap->gp_offset; ap->gp_offset = ap->gp_offset + 8; }",
                "  else { p = (char*)ap->overflow_arg_area; ap->overflow_arg_area = p + 8; }",
                "  return *(long long*)p;",
                "}",
                "int tinyc_va_arg_i32(struct tinyc_va_list_x64 *ap) { return (int)tinyc_va_arg_i64(ap); }",
                "void *tinyc_va_arg_ptr(struct tinyc_va_list_x64 *ap) { return (void*)tinyc_va_arg_i64(ap); }",
                "double tinyc_va_arg_f64(struct tinyc_va_list_x64 *ap) {",
                "  char *p;",
                "  if (ap->fp_offset < 176) { p = (char*)ap->reg_save_area + ap->fp_offset; ap->fp_offset = ap->fp_offset + 16; }",
                "  else { p = (char*)ap->overflow_arg_area; ap->overflow_arg_area = p + 8; }",
                "  return *(double*)p;",
                "}",
                "void tinyc_va_arg_struct(struct tinyc_va_list_x64 *ap, void *out, long long size) {",
                "  long long words; long long i; long long *o;",
                "  words=(size+7)/8; o=(long long*)out;",
                "  for(i=0;i<words;i=i+1){ o[i]=tinyc_va_arg_i64(ap); }",
                "}",
                "",
            ]
        else:
            lines += [
                "int tinyc_va_arg_i32(char **ap) { char *p; int *q; p=*ap; *ap=p+8; q=(int*)p; return *q; }",
                "long long tinyc_va_arg_i64(char **ap) { char *p; long long *q; p=*ap; *ap=p+8; q=(long long*)p; return *q; }",
                "double tinyc_va_arg_f64(char **ap) { char *p; double *q; p=*ap; *ap=p+8; q=(double*)p; return *q; }",
                "void *tinyc_va_arg_ptr(char **ap) { char *p; void **q; p=*ap; *ap=p+8; q=(void**)p; return *q; }",
                "void tinyc_va_arg_struct(char **ap, void *out, long long size) {",
                "  char *p; long long words; long long i; long long *o; long long *s;",
                "  p=*ap; words=(size+7)/8; o=(long long*)out; s=(long long*)p;",
                "  for(i=0;i<words;i=i+1){ o[i]=s[i]; }",
                "  *ap=p+words*8;",
                "}",
                "",
            ]
    if uses_inline_platform():
        prefix = "_" if (TARGET == "macho" or IS_WIN) else ""
        if IS_WIN:
            o_creat = 0x0100
            o_trunc = 0x0200
        else:
            o_creat = 0x0200 if sys.platform.startswith("darwin") else 0x40
            o_trunc = 0x0400 if sys.platform.startswith("darwin") else 0x200
        lines += [
            f"extern long {prefix}write(int fd, const void *buf, unsigned long n);",
            f"extern long {prefix}read(int fd, void *buf, unsigned long n);",
            f"extern int {prefix}open(const char *path, int flags, ...);",
            f"extern int {prefix}close(int fd);",
            f"extern long {prefix}lseek(int fd, long offset, int whence);",
            f"extern void {prefix}exit(int status);",
            "static char tinyc_static_heap_raw[16842752];",
            "void ensure_heap_initialized(void) { }",
            "void platform_init(int argc, char **argv, char **envp) { (void)argc; (void)argv; (void)envp; buddy_init(); }",
            "void *platform_heap_base(void) { unsigned long p = (unsigned long)tinyc_static_heap_raw; p = (p + 65535) & ~65535UL; return (void*)p; }",
            "unsigned long platform_heap_size(void) { return 16777216; }",
            "void *platform_heap_grow(unsigned long n) { (void)n; return (void*)0; }",
            "unsigned int platform_fd_write(int fd, const ciovec_t *iovs, unsigned long iovs_len, unsigned long *nwritten) {",
            "  unsigned long total = 0;",
            f"  for (unsigned long i = 0; i < iovs_len; i = i + 1) total = total + (unsigned long){prefix}write(fd, iovs[i].buf, iovs[i].buf_len);",
            "  *nwritten = total; return 0;",
            "}",
            f"void platform_exit(int status) {{ {prefix}exit(status); }}",
            "int platform_args_sizes_get(unsigned long *argc, unsigned long *argv_buf_size) { *argc = 0; *argv_buf_size = 0; return 0; }",
            "int platform_args_get(char **argv, char *argv_buf) { (void)argv; (void)argv_buf; return 0; }",
            "int platform_environ_sizes_get(unsigned long *n, unsigned long *s) { *n = 0; *s = 0; return 0; }",
            "int platform_environ_get(char **e, char *b) { (void)e; (void)b; return 0; }",
            "int platform_path_open(const char *path, unsigned long path_len, unsigned long long rights, int oflags) {",
            "  int flags = 0;",
            "  int has_read = (rights & 2) != 0;",
            "  int has_write = (rights & 64) != 0;",
            "  if (has_read && has_write) flags = 2; else if (has_write) flags = 1;",
            f"  if (oflags & 1) flags = flags | {o_creat};",
            f"  if (oflags & 8) flags = flags | {o_trunc};",
            f"  (void)path_len; return {prefix}open(path, flags, 0644);",
            "}",
            f"int platform_fd_close(int fd) {{ return {prefix}close(fd); }}",
            "int platform_fd_read(int fd, const iovec_t *iovs, unsigned long iovs_len, unsigned long *nread) {",
            "  unsigned long total = 0;",
            f"  for (unsigned long i = 0; i < iovs_len; i = i + 1) total = total + (unsigned long){prefix}read(fd, iovs[i].iov_base, iovs[i].iov_len);",
            "  *nread = total; return 0;",
            "}",
            f"int platform_fd_seek(int fd, long long offset, int whence, unsigned long long *newoffset) {{ long r = {prefix}lseek(fd, (long)offset, whence); *newoffset = (unsigned long long)r; return r < 0; }}",
            f"int platform_fd_tell(int fd, unsigned long long *offset) {{ long r = {prefix}lseek(fd, 0, 1); *offset = (unsigned long long)r; return r < 0; }}",
            "int platform_read_file_mmap(const char *filename, unsigned long long *out_handle, void **out_data, unsigned long *out_size) { (void)filename; *out_handle = 0; *out_data = (void*)0; *out_size = 0; return 0; }",
            "void platform_file_unmap(unsigned long long handle) { (void)handle; }",
            "double fast_sqrt(double x) { return __builtin_sqrt(x); }",
            "float fast_sqrtf(float x) { return __builtin_sqrtf(x); }",
        ]
    else:
        lines.append(f'#include "{_inc(platform_source_for_unity())}"')
    if host_main:
        if uses_wasm_runtime():
            lines += [
                "",
                "int main(void) {",
                "    platform_init(0, 0, 0);",
                "    return app_main();",
                "}",
            ]
        elif TARGET == "macho" and MACHO_BACKEND == "llvm":
            lines += [
                "",
                "int main(void) {",
                "    platform_init(0, 0, 0);",
                "    return app_main();",
                "}",
            ]
        elif IS_WIN:
            lines += [
                "",
                "int main(int argc, char **argv) {",
                "    platform_init(argc, argv, 0);",
                f"    return {'app_main(argc, argv)' if main_takes_args(srcs) else 'app_main()'};",
                "}",
            ]
        else:
            lines += [
                "",
                "int main(int argc, char **argv, char **envp) {",
                "    platform_init(argc, argv, envp);",
                f"    return {'app_main(argc, argv)' if main_takes_args(srcs) else 'app_main()'};",
                "}",
            ]
    unity.write_text("\n".join(lines) + "\n")
    return unity


def unity_include_flags() -> list[str]:
    return [
        "-I", str(COREC_STDLIB_DIR / "stdlib"),
        "-I", str(COREC_STDLIB_COREC_DIR),
        "-I", str(HERE / "tests"),
    ]


def link_native(obj_path: Path, exe_path: Path):
    """Link the llc-produced single-TU object."""
    if IS_WIN:
        # MSVC: cl /nologo /MD obj /Fe:exe.exe
        return run([
            CC, "/nologo", "/MD",
            str(obj_path), str(HERE / "tinyc_wasm_vararg.c"),
            f"/Fe:{exe_path}",
        ])
    cmd = [CC, str(obj_path), "-o", str(exe_path)]
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
            "-DTINYC_WASM_RUNTIME_NO_LIBC",
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
    runtime_args = []
    if runtime_obj is not None:
        runtime_args.append(str(runtime_obj))
    if start_obj is not None:
        runtime_args.append(str(start_obj))
    if USE_NATIVE_LINK:
        return run([
            str(NATIVE_LINK), "--link",
            "-o", str(wasm_path),
            "--export=_start",
            *obj_args, *runtime_args,
        ])
    return run([
        WASM_LD, "--no-entry", "--export=_start",
        *obj_args, *runtime_args,
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

    # Pre-build the tinyC wasm runtime object once per run. The generated
    # single-TU root contains corec + stdlib + the test; runtime_wasm.c still
    # supplies tinyC's lowered print/va_arg hooks and the WASI _start shim.
    wasm_runtime_obj = HERE / "tests" / "runtime_wasm.o"
    wasm_start_obj   = HERE / "tests" / "start_wasm.o"
    if uses_wasm_runtime():
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
        if name == "func_macro" and not use_unity_source():
            expected = "greet\nsquare=16\nmain\n"
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
        if t.get("link_separately"):
            print(f"SKIP {name} (link_separately; not a single-TU test)")
            skipped += 1
            continue
        if t.get("single_tu_skip"):
            print(f"SKIP {name} (single_tu_skip)")
            skipped += 1
            continue
        # Multi-file tests pass `sources = [...]`; single-file tests
        # default to `<name>.tc` for backwards compatibility.
        sources = t.get("sources", [f"{name}.tc"])
        srcs = [HERE / "tests" / s for s in sources]
        unity_src = write_unity_source(name, srcs) if use_unity_source() else None

        if TARGET == "wasm":
            obj  = HERE / "tests" / f"{name}.wasm.o"
            wasm = HERE / "tests" / f"{name}.wasm"

            # Stage 1: tinyc emits one wasm32 object from the generated
            # single-TU root.
            r = run([str(TINYC), "--emit=wasm", *LOWERING_FLAG,
                     *unity_include_flags(),
                     "-o", str(obj),
                     str(unity_src)])
            if r.returncode != 0:
                print(f"FAIL {name}: tinyc returned {r.returncode}\nstderr:\n{r.stderr}")
                failures += 1
                continue

            # Stage 2: link with runtime_wasm.o + start_wasm.o for tinyC's
            # lowered print/va_arg hooks and the WASI entry shim.
            r = link_wasm([obj], wasm_runtime_obj, wasm_start_obj, wasm)
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
                         *unity_include_flags(),
                         "-o", str(exe)]
            tinyc_cmd.append(str(unity_src))
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
                obj_j = HERE / "tests" / f"{name}.macho.wasm.o"
                r = run([str(TINYC), "--emit=wasm", *LOWERING_FLAG,
                         *unity_include_flags(),
                         "-o", str(obj_j),
                         str(unity_src)])
                if r.returncode != 0:
                    print(f"FAIL {name}: tinyc --emit=wasm returned {r.returncode}\nstderr:\n{r.stderr}")
                    failures += 1
                    continue
                linked_wasm = HERE / "tests" / f"{name}.macho.linked.wasm"
                r = link_wasm([obj_j], wasm_runtime_obj, wasm_start_obj, linked_wasm)
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
                         *unity_include_flags(),
                         "-o", str(exe)]
            if MACHO_BACKEND == "llvm":
                tinyc_cmd.append("--macho-backend=llvm")
            else:
                tinyc_cmd.append(f"--wasm-runtime-obj={wasm_runtime_obj}")
                tinyc_cmd.append(f"--wasm-runtime-obj={wasm_start_obj}")
            tinyc_cmd.append(str(unity_src))
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

        # Stage 1: emit LLVM IR from the generated single-TU root.
        if unity_src is not None:
            tinyc_llvm_cmd = [str(TINYC), "--emit=llvm", *LOWERING_FLAG,
                              *unity_include_flags(), str(unity_src)]
        else:
            tinyc_llvm_cmd = [str(TINYC), "--emit=llvm", *LOWERING_FLAG,
                              *unity_include_flags(),
                              *[str(s) for s in srcs]]
        r = run(tinyc_llvm_cmd)
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
        # then link with the platform compiler.
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

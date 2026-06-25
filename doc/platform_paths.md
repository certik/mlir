# tinyC compilation paths and the platform layer

How a tinyC-compiled program reaches the operating system, across every
backend, and where (if anywhere) that bypasses corec's platform layer.

This is the result of issue #204 ("Refactoring runtime"): removing the
hand-written "magic runtime" so that, with two well-understood test-harness
exceptions, **all OS access goes through corec's `platform.h`**.

## The platform contract

Every tinyC-compiled program reaches the OS through one interface,
`corec/platform/platform.h` (~20 functions):

- I/O: `platform_fd_write`, `platform_fd_read`, `platform_fd_close`,
  `platform_fd_seek`, `platform_fd_tell`, `platform_path_open`
- memory: `platform_heap_base`, `platform_heap_size`, `platform_heap_grow`
- process: `platform_init`, `platform_exit`
- args/env: `platform_args_sizes_get`, `platform_args_get`,
  `platform_environ_sizes_get`, `platform_environ_get`
- misc: `platform_read_file_mmap`, `platform_file_unmap`, `fast_sqrt`,
  `fast_sqrtf`

`corec-stdlib` (`printf`, `malloc`, the `FILE*` layer, …) is written entirely
against this interface — it never talks to the OS directly. Four
implementations of the contract exist, one per platform:

| implementation | how it reaches the OS |
| --- | --- |
| `corec/platform/platform_linux.c`   | raw syscalls via `__builtin_syscall6` |
| `corec/platform/platform_macos.c`   | libc (`writev` / `readv` / `mmap` / `mprotect` / …) |
| `corec/platform/platform_windows.c` | WinAPI (`__declspec(dllimport)` / `__stdcall`) |
| `corec/platform/platform_wasm.c`    | the eleven WASI imports |

A program is therefore always: **user C + corec-stdlib + one
`platform_<os>.c`**, compiled together. The three compilation paths below
differ only in *how* the chosen `platform_<os>.c` is brought in and lowered.

## Path 1 — Native backends

C → MLIR → the in-house `llvm` dialect → machine code. The unity source
`#include`s the real `platform_<os>.c`, so the platform implementation is
compiled into the same module as the program.

Three sub-flavors:

- **ELF / x86_64 — `--emit=elf`** → `mlir_llvm_to_elf`.
  Unity `#include`s `platform_linux.c`. The backend itself synthesizes the raw
  `syscall` thunk for `__tinyc_syscall6` (`mlir_llvm_to_x64_emit.inc`,
  `synth_thunk`). Output: a complete static ELF executable, no libc, no
  dynamic linker.
- **Mach-O / arm64 — `--emit=macho --macho-backend=llvm`** →
  `mlir_llvm_to_aarch64` → `mlir_aarch64_to_macho`.
  Unity `#include`s `platform_macos.c`; its libc calls (`writev`, `mmap`,
  `mprotect`, …) resolve to libSystem stubs in the Mach-O import table.
  Output: a complete Mach-O executable.
- **LLVM-IR / llc — `--emit=llvm` → `llc` → host link** (the harness's
  `native` target). Unity `#include`s the real `platform_<os>.c`. On Linux the
  driver rewrites the emitted IR to define `@__tinyc_syscall6` as an x86_64
  `syscall` inline-asm (`driver.c`, `inject_syscall6_definition`), so
  `platform_linux.c`'s raw syscalls link through the host toolchain.

All three go **via corec**, except the Windows `native` case — see
[Exceptions](#does-it-always-go-via-corec).

## Path 2 — Wasm

C → MLIR → wasm. `--emit=wasm`, then `tinyc --link` produces a WASI module.

The unity `#include`s **`platform_wasm.c`**, which *is* corec's platform
implementation for wasm: it declares the eleven WASI imports
(`fd_write` / `proc_exit` / `path_open` / …) with
`__attribute__((__import_module__("wasi_snapshot_preview1")))` and provides the
real `_start → platform_init_and_run → app_main`. The actual I/O is performed
by the **WASI host** (wasmtime, or the browser shim) that instantiates the
module.

Goes **via corec** (`platform_wasm.c`) → WASI host.

## Path 3 — Wasm → native (lift)

A linked `.wasm` is lifted back to native code:
`--from-wasm linked.wasm --emit=macho|elf --host-platform=platform_<os>.c
--wasi-adapter=corec/wasm/wasi_adapter.c`
(wasm → wasmstack → wasmssa → `llvm` dialect → native).

The lifted program still *calls* the WASI imports, but their bodies are **no
longer synthesized by the compiler**. Instead the driver compiles and splices
in two real corec C files (`tinyc_compile_host_platform` in `driver.c`):

```
WASI import (fd_write, path_open, …)
  → corec/wasm/wasi_adapter.c     translate wasm32 linmem offsets → host
                                  pointers, repack iovecs, store results back
    → __host_platform_*           = corec/platform/platform_<os>.c, compiled
                                    LP64 with platform_* renamed to
                                    __host_platform_*
      → OS
```

(`proc_exit` is the one exception inside this path: it carries no linmem-offset
arguments, so the lifter maps it straight to `__host_platform_exit` rather than
routing it through the adapter.)

Goes **via corec** — both the WASI adapter and the platform implementation are
ordinary corec C, compiled by tinyC.

## Does it always go via corec?

**Yes, except for two test-harness accommodations** in
`examples/tinyc/run_tinyc_tests.py`. They are properties of how the *test
runner* builds programs, not of the compiler — the compiler always emits
`platform_*` / WASI calls and binds them to corec implementations when given
them.

1. **Windows `native`** (gated by `use_unity_source()` returning `False` for
   `IS_WIN and TARGET == "native"`). The harness compiles each test `.tc`
   *directly* against the MSVCRT, with no unity, no corec-stdlib, and no
   `platform_windows.c`. The test's `printf` binds to MSVCRT's. Reason: tinyC
   cannot yet parse `platform_windows.c`'s `__declspec(dllimport)` /
   `__stdcall`, and the suite only needs libc-providable symbols. So Windows
   `native` is both runtime-shim-free *and* corec-free on this path.
2. **`macho-llvm` cross-compiled on a non-Apple host** (e.g. Linux building a
   Mach-O; gated by `uses_inline_platform()`). The harness substitutes a small
   inline platform shim (a ~45-line block synthesized as Python strings)
   instead of `platform_macos.c`, because libSystem cannot be linked off-Apple.
   Pure cross-compilation artifact; on an actual macOS host this target uses
   the real `platform_macos.c`.

Everything else — every backend, on its native host — reaches the OS through
`corec/platform/platform_<os>.c`.

## Deferred (tracked, not "magic runtime")

- **aarch64 `synth_start`** (`mlir_llvm_to_aarch64.c`): the wasm-lifted→native
  crt0/loader that allocates and copies the linear-memory image and pins
  `x28` = linmem base / `x27` = globals base before calling `main`. It is
  native-coupled (PC-relative relocations + reserved-register init) and not
  cleanly expressible as portable C, so it stays a compiler internal.
- **Compiling `platform_windows.c` with tinyC**: would remove exception (1)
  above, but needs tinyC parser support for `__declspec(dllimport)` /
  `__stdcall`. Optional consistency work; Windows is already shim-free.

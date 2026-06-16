# Native backends and the WebAssembly linear-memory model

This document explains the two ways tinyC code reaches a native binary
(x86_64/ELF and aarch64/Mach-O), the WebAssembly **linear-memory model** that the
"via-wasm" path carries, and how invasive that model is to a backend. It exists
to make an explicit architectural decision visible: the x86_64/ELF backend takes
the **direct** path and does not (currently) implement the linear-memory model.

## Two paths from C to a native binary

A tinyC program can be lowered to a native binary two different ways. Both end in
the in-house `llvm` dialect and one of the native code generators
(`mlir_llvm_to_x64.c` → ELF, `mlir_llvm_to_aarch64.c` → Mach-O), but they differ
in *what the `llvm` dialect looks like* by the time it gets there.

### 1. Direct path  (C → LLVM → native, LP64)

```
foo.tc ──► tinyC front end ──► llvm dialect (LP64) ──► mlir_llvm_to_{x64,aarch64} ──► ELF / Mach-O
```

* Native data model: **LP64** — 64-bit pointers that are *real machine
  addresses*, 64-bit `long`.
* Memory ops use absolute addressing: `llvm.load %p` → `mov rax, [rax]`.
* This is what `--emit=elf` and `--macho-backend=llvm` use today. The x86_64/ELF
  backend supports **only** this path.

### 2. Via-wasm path  (C → wasm → lift → LLVM → native)

```
foo.tc ─► --emit=wasm ─► wasm-ld (links many .wasm.o) ─► --from-wasm ─► lift wasm→llvm ─► native
```

* Used by the default `--emit=macho` backend and `--macho-backend=llvm_via_wasm`.
* Its big advantage is **free multi-TU linking and variadics**: `wasm-ld` resolves
  `static` / `weak` / duplicate symbols across many object files, and the lifter
  (`mlir_wasmssa_to_llvm.c`) turns `va_arg` into synthesized `tinyc_va_arg_*`
  helpers — so the native backend never needs its own linker or `va_start`
  lowering.
* Its cost is that it carries the WebAssembly **linear-memory model** all the way
  into native code generation.

## The linear-memory model

WebAssembly has exactly **one flat byte array** for all of its memory. Every wasm
pointer is a **32-bit offset** into that array; there are no native addresses at
the wasm level. A wasm `load p` means "read `linmem[p]`", i.e. native address
`linmem_base + p`.

When wasm is lifted back to the `llvm` dialect, this is made **explicit in the
IR**:

* A global **`__wasm_linmem_base`** (an `i64`) holds the real base address of the
  emulated linear memory. Every memory access becomes
  `load(@__wasm_linmem_base) + offset` address arithmetic.
* **`__wasm_linmem`** is the *template*: the initial data segment (string
  literals, initialized globals) that must be copied into linear memory at
  startup.
* Escaping locals live on a **wasm shadow stack** — a region *inside* linear
  memory pointed at by a wasm global (`__wasm_g0`), **not** the native `sp`/`rbp`.
* Other wasm globals (argc/argv, the shadow-stack pointer, …) live in a small
  **globals cluster** in the data section.

So the lifted program does not use native pointers for its heap/stack data at
all — it uses 32-bit offsets relative to a base the runtime sets up.

## How a backend implements it (aarch64, as the reference)

`mlir_llvm_to_aarch64.c` implements the model with three discrete pieces, all
**gated on the presence of `__wasm_linmem_base`**:

1. **A reserved base register.** `x28` is pinned to `linmem_base` for the entire
   program and taken out of the allocatable pool; `x27` likewise anchors the
   wasm-globals cluster (`mlir_llvm_to_aarch64.c:2511`). Because they live outside
   the register pool, every function treats them as callee-saved and the base
   flows through the whole call tree with no reload.

2. **A custom `synth_start`** (`mlir_llvm_to_aarch64.c:3673`). When
   `__wasm_linmem_base` exists it: `mmap`s the linear memory (4 GiB reserved),
   `memcpy`s the `__wasm_linmem` template into it, stores the base into
   `__wasm_linmem_base`, sets `x28 = base`, initializes the wasm globals
   (shadow-stack pointer, argc/argv), and only then calls `main`.

3. **Base-pinning in the lifted lowering path** (`select_func_cfg`,
   `mlir_llvm_to_aarch64.c:3093`). It recognizes the `__wasm_linmem_base` load
   pattern (`is_linmem_base_load`), replaces it with the pinned register, and
   lowers `base + offset` as register-offset addressing (`ldr [x28, xoff]`).

## Is it additive, or does it change all lowering?

**Mostly additive — it is a parallel path, not a rewrite.** The dispatch is
explicit (`select_func`, `mlir_llvm_to_aarch64.c:3568`):

```c
if (RegionNumBlocks > 1)          // flat multi-block CFG == wasm-lifted
    return select_func_cfg(...);  // linmem-aware driver
// else: single-block structured scf == the direct C→LLVM path
```

Two drivers share the per-op helpers (`lower_op` for arithmetic / calls). The
direct path never emits `__wasm_linmem_base` or wasm globals, so:

* `synth_start`'s linmem setup is skipped (`has_linmem == false`),
* base-pinning never fires,
* absolute native addressing is used.

The existing direct path is therefore **untouched** by the model. What *is*
pervasive is small and discrete:

* **One or two globally reserved registers** (the direct path also won't allocate
  `x27`/`x28`).
* **The load/store address formation gains a linmem-aware branch** (pinned base
  vs absolute). That is the only place "all memory lowering" is conceptually
  aware of linmem, and it is a gated branch, not a rewrite.

Rough split: ~85% additive (a parallel driver + a custom `synth_start`), ~15%
shared cost (one reserved register + a linmem branch in address formation).

## Status and decision for x86_64/ELF

`mlir_llvm_to_x64.c` / `mlir_llvm_to_x64_emit.inc` implement **only the direct
path** and have **zero** linear-memory handling (aarch64 has ~46 references).
A prototype `--from-wasm --emit=elf` lifts and runs — `simple.tc` even prints the
right value — but then crashes (`SIGBUS`) because the shadow stack the lifted code
relies on is never initialized.

For x86_64/ELF we therefore **commit to the direct path** rather than porting the
linear-memory model:

* It reuses the already-working LP64 backend (no new register pressure, no
  wasm32 semantics, no 4 GiB `mmap`, native pointers throughout).
* Multi-TU linking is handled instead by compiling **everything as one big
  translation unit** (the test + corec + corec-stdlib + `platform_<os>.c`), with
  the front end giving `static` symbols internal-linkage scoping and de-duplicating
  `weak` definitions — no native linker required.
* The remaining direct-path gap is native variadics (`llvm.intr.vastart` /
  `va_arg`), mirroring the aarch64 lowering.

The via-wasm path (and its linear-memory model) remains the right tool for the
aarch64/Mach-O backend, where it is already implemented and exercised by the
self-host.

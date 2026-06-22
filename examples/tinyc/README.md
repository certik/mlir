# tinyC — a tiny C-subset compiler built on the C MLIR API

This is the first end-to-end compiler example for this repository. It
compiles a small subset of C straight to a native binary using only the
public C API in `mlir_api.h` (plus the corec primitives — arena, string,
io, platform). It does **not** include any upstream MLIR header
directly.

The pipeline:

```
.tc source
   │  examples/tinyc/lex.c    (hand-rolled lexer)
   ▼
tokens
   │  examples/tinyc/parse.c  (recursive descent)
   ▼
AST
   │  examples/tinyc/emit.c   (uses mlir_api.h ONLY)
   ▼
builtin.module IR  (func + arith + memref + scf + vector dialects)
   │  MLIR_LowerToLLVMDialect    (new in mlir_api.h)
   ▼
LLVM dialect IR
   │  MLIR_TranslateModuleToLLVMIR
   ▼
LLVM IR text
   │  $CC out.ll -o a.out
   ▼
native binary
```

## Language subset

tinyC has grown well past the initial "int-only" prototype. The `tests/`
directory is the working capability matrix; in broad strokes the
frontend now supports:

- Types: `int`, `char`, signed/unsigned variants, `float`, `double`,
  pointers, fixed-size arrays, `struct`/`union`, `typedef`.
- Locals and globals, with zero / designated / compound initializers.
- All the usual operators (`+ - * / %`, comparisons, `&& || !`, bitwise,
  unary `-`, `sizeof`, `& *`, postfix/prefix `++ --`).
- Statements: assignment, `if`/`else`, `while`, `do`/`while`, `for`,
  `break`, `continue`, early `return`, `switch`/`case`/`default`,
  blocks `{ … }`, expression statements.
- Functions: fixed-arity (`int name(int p, …)`) and variadic
  (`int sum(int n, ...)` — emitted as `llvm.func` since `func.func`
  has no var-arg form).
- A generated single-TU test root that supplies corec/corec-stdlib for
  native tests, plus a wasm runtime object for wasm tests.
- A `#define` / `#include` preprocessor (`preprocess.c`), with `-I`
  include dirs and `-D` definitions on the CLI.

The wasm target is a strict subset of the native one — a handful of
tests are tagged `wasm_skip` because they rely on host-libc behaviour
wasm32-wasi cannot reproduce.

## Build & run

```sh
pixi run build_tinyc_upstream
pixi run test_tinyc_upstream
```

To inspect the IR at any pipeline stage:

```sh
./tinyc                    examples/tinyc/tests/sum_to_10.tc   # high-level MLIR
./tinyc --emit=lowered     examples/tinyc/tests/sum_to_10.tc   # LLVM dialect
./tinyc --emit=llvm        examples/tinyc/tests/sum_to_10.tc   # LLVM IR (.ll)
./tinyc --emit=wasm -o /tmp/p.wasm.o examples/tinyc/tests/sum_to_10.tc  # wasm32 object
```

To produce a binary by hand (the test runner does this for you):

```sh
./tinyc --emit=llvm examples/tinyc/tests/sum_to_10.tc > /tmp/p.ll
$CC /tmp/p.ll -o /tmp/p
/tmp/p   # prints 55
```

## WebAssembly backend

`tinyc` can emit a wasm32 (WASI) object via `--emit=wasm`. There are two
independent axes you can vary:

| Flag                  | Choice              | What it controls                                |
| --------------------- | ------------------- | ----------------------------------------------- |
| `--lowering=upstream` | (default)           | MLIR `llvm`/`scf`/`cf` → LLVM IR via upstream MLIR conversion passes, then LLVM's WebAssembly backend. |
| `--lowering=native`   |                     | MLIR `llvm` dialect → wasm32 directly via the in-tree three-stage pipeline (no LLVM backend on the WASM side). |

The two paths produce equivalent `.wasm.o` objects which are then linked
with `wasm-ld` and run under `wasmtime`. Both pass the full tinyC test
suite (150 wasm tests on Linux/macOS; 11 are skipped under wasm for
unrelated reasons).

### Compile and run a single program

A standalone program links against the corec runtime prelude
(`examples/tinyc/browser/prelude.c` — `printf` / `malloc` / the WASI
platform layer + the `_start` entry). tinyC lowers `va_arg` inline, so no
external va_arg object is needed. Build the prelude once:

```sh
./tinyc --emit=wasm -I corec-stdlib/stdlib -I corec-stdlib/corec \
    -o /tmp/corec_runtime.wasm.o examples/tinyc/browser/prelude.c
```

Then compile and link a program. `_tinyc_print` lowers to a real `printf`
call once `printf` is declared, and `#define main app_main` lets the
prelude's `_start` reach the program entry:

```sh
{ printf '#define main app_main\nextern int printf(const char*,...);\n'; \
  cat examples/tinyc/tests/sum_to_10.tc; } > /tmp/p.tc

# Upstream lowering (LLVM's WASM backend)
./tinyc --emit=wasm --lowering=upstream -o /tmp/p.wasm.o /tmp/p.tc
wasm-ld --no-entry --export=_start --allow-undefined \
    /tmp/p.wasm.o /tmp/corec_runtime.wasm.o \
    -o /tmp/p.wasm
wasmtime /tmp/p.wasm   # prints 55

# Native lowering (in-tree wasmssa → wasmstack → binary pipeline)
./tinyc --emit=wasm --lowering=native -o /tmp/p.wasm.o /tmp/p.tc
# (link + run identical to above)
```

### Inspect intermediate stages

```sh
./tinyc --emit=mlir       sum_to_10.tc     # 1. high-level MLIR (funcs in mixed dialects)
./tinyc --emit=lowered    sum_to_10.tc     # 2. lowered to LLVM dialect
./tinyc --emit=llvm       sum_to_10.tc     # 3. LLVM IR text
./tinyc --emit=wasmssa  --lowering=native sum_to_10.tc   # 4. native pipeline stage 1
./tinyc --emit=wasmstack --lowering=native sum_to_10.tc  # 5. native pipeline stage 2
./tinyc --emit=wat        sum_to_10.tc     # 6. final wasm binary as WAT
```

`--emit=wasmssa` and `--emit=wasmstack` print the intermediate
wasmssa / wasmstack MLIR modules used by the native pipeline in the
generic MLIR text format; both require `--lowering=native`.
`--emit=wat` disassembles the final `.wasm.o` bytes to a WAT-like text
form and works with either lowering.

### Run the test suite

The four supported (lowering × target) combinations and the pixi tasks
that drive them:

| Lowering | Target | pixi command |
| -------- | ------ | ------------ |
| upstream | native | `pixi run -e upstream test_tinyc_upstream` |
| native   | native | `TINYC_LOWERING=native pixi run -e upstream test_tinyc_upstream` |
| upstream | wasm   | `pixi run -e upstream test_tinyc_upstream_wasm` |
| native   | wasm   | `TINYC_LOWERING=native pixi run -e upstream test_tinyc_upstream_wasm` |

`TINYC_LOWERING` (default `upstream`) selects the lowering used by the
test runner; `TINYC_TARGET` (default `native`) selects the target. The
`*_wasm` task simply sets `TINYC_TARGET=wasm`.

The `tinyc_native` binary (built without upstream MLIR/LLVM, via
`pixi run build_tinyc_native`) supports both the native and wasm
targets — the wasm pipeline is entirely in-tree. Use
`pixi run test_tinyc_native` for the native target and
`pixi run test_tinyc_native_wasm` for the wasm target.

### Self-host under wasmtime

`tinyc.wasm` is built once with clang (`pixi run build_tinyc_wasm`),
then under wasmtime it compiles its own source set into a stage-2
`tinyc.wasm`, and the stage-2 module re-compiles the same source set
into a stage-3 module. A bit-identical comparison of the stage-2 and
stage-3 binaries is the self-hosting fixed-point check:

```sh
pixi run -e wasm verify_tinyc_wasm_selfhost
# tinyc.wasm self-host is bit-identical (stage2 == stage3)
```

The individual steps (`selfhost_tinyc_wasm_stage2`,
`selfhost_tinyc_wasm_stage3`) are also available as standalone pixi
tasks. Each stage drops its per-file `.wasm.o` outputs under
`selfhost_stage2/` / `selfhost_stage3/` for inspection.

### How the native pipeline is structured

The cf→scf lift is performed up front by `MLIR_LowerToLLVMDialectForWasm`
(`mlir_lower_to_llvm.c`, calling `MLIR_LiftCfToScf` →
`mlir_lift_cf_to_scf.c`) so the wasm stages see already-structured
control flow. The LLVM-dialect → wasm path then runs three MLIR-to-MLIR
stages:

- `mlir_llvm_to_wasmssa.c` — Stage 1: LLVM dialect → wasmssa
  (lowers ops in-place; the cf→scf lift has already happened above).
- `mlir_wasmssa_to_wasmstack.c` — Stage 2: wasmssa → wasmstack (linearises
  SSA results onto the wasm value stack, inserts block/loop/if/end).
- `mlir_wasmstack_to_bin.c` — Stage 3: wasmstack → `.wasm.o`
  (emits the wasm binary including type/import/function/table/memory/
  global/export/element/code/data sections plus relocation metadata
  consumed by `wasm-ld`).
- `mlir_wasm_to_wat.c` — generic-MLIR printer used by `--emit=wasmssa`
  and `--emit=wasmstack`, plus a `.wasm` → WAT disassembler.

`MLIR_TranslateModuleToWasm` (in `mlir_translate_to_wasm.c`) chains the
three stages and is linked into both the native- and upstream-backed
tinyc binaries.

## Why this exists

This example is the working proof that the C MLIR API in `mlir_api.h`
is rich enough to serve as a full replacement for upstream MLIR's C++
API for compiler authors. Writing a real frontend exposed several gaps
that were filled in `mlir_api.h` along the way:

- `MLIR_CreateAttributeSymbolRef` — needed by `func.call` (a `StringAttr`
  is rejected by the verifier; you need a real `FlatSymbolRefAttr`).
- `MLIR_LowerToLLVMDialect` — runs the standard MLIR conversion passes
  (`scf-to-cf`, `*-to-llvm`, `reconcile-unrealized-casts`).
- `MLIR_LowerToLLVMDialectForWasm` + `MLIR_LiftCfToScf` — the wasm
  variant that lifts cf→scf first and keeps `scf.*` ops in place.
- `MLIR_TranslateModuleToLLVMIR` — emits portable LLVM IR text.
- `MLIR_TranslateModuleToWasm` — emits a wasm32-wasi object file via
  the three-stage in-tree pipeline.

Both backends implement all of the above. The native backend's
lowering, lift, and wasm pipeline are written in plain C against
`mlir_api.h` only (see `mlir_lower_to_llvm.c`, `mlir_lift_cf_to_scf.c`,
`mlir_translate_to_wasm.c`, and the three `mlir_*_to_wasm*` stages),
so `tinyc_native` produces functionally equivalent output without
linking any LLVM/MLIR libraries.

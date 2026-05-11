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
   │  $CC out.ll runtime.c -o a.out
   ▼
native binary
```

## Language subset

- Types: `int` only (i32).
- Locals: `int x;` and `int x = expr;` — stored in `memref<i32>`.
- Operators: `+ - * / %`, `< <= > >= == !=`, `&& || !`, unary `-`.
- Statements: assignment, `if`/`else`, `while`, `return`, `_tinyc_print(expr);`,
  blocks `{ … }`. `print` lowers to `vector.print`.
- Functions: `int name(int p, int q, …) { … }` returning int.
- `main()` is the program entry point.

Notable limitations (kept minimal on purpose):
- No early `return` from inside `if`/`while` — use a result variable.
- No arrays, pointers, structs, floats, chars, strings, `for`,
  `break`/`continue`.

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
$CC /tmp/p.ll examples/tinyc/runtime.c -o /tmp/p
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

```sh
# Upstream lowering (LLVM's WASM backend)
./tinyc --emit=wasm --lowering=upstream \
    -o /tmp/p.wasm.o examples/tinyc/tests/sum_to_10.tc
wasm-ld --no-entry --export=_start --allow-undefined \
    /tmp/p.wasm.o examples/tinyc/runtime_wasm.o examples/tinyc/start_wasm.o \
    -o /tmp/p.wasm
wasmtime /tmp/p.wasm   # prints 55

# Native lowering (in-tree wasmssa → wasmstack → binary pipeline)
./tinyc --emit=wasm --lowering=native \
    -o /tmp/p.wasm.o examples/tinyc/tests/sum_to_10.tc
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
`pixi run build_tinyc_native`) currently supports the native target
only; for that binary use `pixi run test_tinyc_native`.

### How the native pipeline is structured

The native LLVM-dialect → wasm path lives in three MLIR-to-MLIR stages
plus a thin façade:

- `mlir_lower_llvm_to_wasmssa.c` — Stage 1: LLVM dialect → wasmssa
  (also runs `lift-cf-to-scf` first to recover structured control
  flow from `cf.br`/`cf.cond_br`).
- `mlir_stackify_wasmssa.c` — Stage 2: wasmssa → wasmstack (linearises
  SSA results onto the wasm value stack, inserts block/loop/if/end).
- `mlir_translate_wasmstack_to_binary.c` — Stage 3: wasmstack → `.wasm.o`
  (emits the wasm binary including type/import/function/table/memory/
  global/export/element/code/data sections plus relocation metadata
  consumed by `wasm-ld`).
- `mlir_wasm_print.c` — generic-MLIR printer used by `--emit=wasmssa`
  and `--emit=wasmstack`.
- `mlir_translate_to_wasm.c` — façade chaining the three stages.

## Why this exists

This example is the working proof that the C MLIR API in `mlir_api.h`
is rich enough to serve as a full replacement for upstream MLIR's C++
API for compiler authors. Writing a real frontend exposed several gaps
that were filled in `mlir_api.h` along the way:

- `MLIR_CreateAttributeSymbolRef` — needed by `func.call` (a `StringAttr`
  is rejected by the verifier; you need a real `FlatSymbolRefAttr`).
- `MLIR_LowerToLLVMDialect` — runs the standard MLIR conversion passes
  (`scf-to-cf`, `*-to-llvm`, `reconcile-unrealized-casts`).
- `MLIR_TranslateModuleToLLVMIR` — emits portable LLVM IR text.

The native backend stubs the lowering APIs (returns false / empty
string) and accepts `MLIR_CreateAttributeSymbolRef` by storing the
symbol name as a plain string attribute — sufficient for printing.

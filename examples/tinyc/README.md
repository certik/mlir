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
- Statements: assignment, `if`/`else`, `while`, `return`, `print(expr);`,
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
```

To produce a binary by hand (the test runner does this for you):

```sh
./tinyc --emit=llvm examples/tinyc/tests/sum_to_10.tc > /tmp/p.ll
$CC /tmp/p.ll examples/tinyc/runtime.c -o /tmp/p
/tmp/p   # prints 55
```

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

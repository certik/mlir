# mlir — a small MLIR core in C

A compact [MLIR](https://mlir.llvm.org/) core written in C, built on top
of [Core C](https://github.com/certik/corec), and complementary to
upstream LLVM/MLIR. A single C API, `mlir_api.h`, has two interchangeable
implementations behind it:

* **native** — pure C, no LLVM/MLIR dependency, builds with
  `-nostdlib -nostdinc -fno-builtin`. Quick to build, easy to embed,
  portable down to WebAssembly.
* **upstream** — thin C++ shim that delegates to real `mlir::Operation`,
  `mlir::Type`, etc., so a compiler written against this API can use
  upstream LLVM/MLIR directly.

A compiler written once against `mlir_api.h` can be linked against either
backend without code changes. Both backends are required to produce
byte-identical textual output for every input the test suite exercises,
so the two are kept in lockstep by construction.

Why use it:

* **Tiny build, no LLVM dependency** on the native side — drop it into
  another project, ship a small CLI, or run it inside a WASI sandbox.
* **Fast iteration.** Hand-written tokenizer, recursive-descent parser,
  arena allocation, no C runtime — clean builds are seconds, not minutes.
* **Same API, two backends.** Develop against the native build for speed,
  link against the upstream backend when you want it; the compiler code
  is the same either way.
* **Cross-validated against upstream MLIR.** Every classic-form reference
  must round-trip through the real upstream parser, and native and
  upstream output is diffed byte-for-byte in CI.
* **Portable.** One source tree builds on Linux, macOS, Windows and
  WebAssembly (WASI).

This repository is built `-nostdlib -nostdinc -fno-builtin` (native side)
with the [`corec`](https://github.com/certik/corec) and
[`corec-stdlib`](https://github.com/certik/corec-stdlib) submodules
providing the platform layer, allocators, strings, formatting and a stdlib
subset. Everything is driven by [pixi](https://pixi.sh) and runs on Linux,
macOS, Windows and WebAssembly (WASI).

## Motivation

A single C API, `mlir_api.h`, with two interchangeable implementations:

1. **A small, narrow C API** for working with operations, blocks, regions,
   types, attributes and values. Two implementations live behind that
   API and are interchangeable from the compiler's point of view:
   * **native** — pure-C, no LLVM/MLIR dependency, builds with `-nostdlib`.
   * **upstream** — thin C++ shim that delegates to real `mlir::Operation`,
     `mlir::Type`, etc., so upstream LLVM/MLIR is a first-class backend.
2. **Cross-validation between the two backends.** Every parser/printer
   combination is exercised by the test runner. Native and upstream are
   required to produce byte-identical output, and every classic-form
   reference must round-trip cleanly through the upstream parser.
3. **No standard library, no C runtime (native side).** The native binary
   inherits Core C's sandbox: a single narrow `platform.h` interface, arena
   allocators, custom `string` / `format` / I/O. The same source compiles
   to native binaries on Linux/macOS/Windows and to a WASI `.wasm`
   artifact, which makes it convenient to embed alongside or on top of an
   existing upstream-based toolchain.

## Getting started

```bash
git clone --recursive https://github.com/certik/mlir.git
cd mlir
```

If you cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

All builds are driven by [pixi](https://pixi.sh) — install once and `pixi`
manages every other dependency (re2c, clang, Python, etc.) inside a per-task
environment.

## Build & test

Pick the environment for your host:

```bash
pixi run -e macos   test_macos       # macOS native (arm64)
pixi run -e linux   test_linux       # Linux native (x86_64)
pixi run -e windows test_windows     # Windows native (MSVC, vcvars64)
pixi run -e wasm    test_wasm        # WebAssembly via wasmtime
```

Each `test_<platform>` task builds the parser binary (`parser`,
`parser.exe`, or `parser.wasm`), runs `./parser --construct`, and then runs
the diff-test suite via `python run_tests.py`.

To build without running tests, use `build_<platform>` instead.

`pixi task list` enumerates every available task with descriptions.

### The upstream backend

A second binary, `parser_upstream`, links against a real LLVM/MLIR install
and provides the trusted oracle for cross-validation. It lives behind its
own pixi feature so the regular build does not depend on LLVM:

```bash
pixi run -e upstream build_parser_upstream    # build parser_upstream
pixi run -e upstream test_parser_upstream     # run tests against it
pixi run -e upstream validate_refs            # round-trip all refs through upstream
```

The cross-impl driver (`build_cross_native_macos` / `build_cross_upstream`,
`test_cross_*`) runs the same C entry point against both implementations
and diffs the textual output.

### Reference files

Each `tests/*.mlir` and `tests/*.ttir` file produces multiple reference
outputs in `tests/reference/`, one per `(parse_kind, print_kind)` combo
exercised by `run_tests.py`:

* `*.classic.classic.out` — classic parser → classic (pretty) printer
* `*.classic.generic.out` — classic parser → generic (`"op.name"(...)`) printer
* `*.upstream.classic.out` — upstream parser → classic printer
* `*.upstream.generic.out` — upstream parser → generic printer
* `*.upstream.upstream.out` — upstream parser → upstream printer

Native and upstream are both required to produce **byte-identical** classic
and generic output for every input.

To regenerate references after an intentional output change:

```bash
pixi run -e upstream python run_tests.py -u
```

CI never runs with `-u` — drift must always be reviewed via `git diff`.

## Continuous Integration

`.github/workflows/CI.yml` runs the full matrix on every push and PR:

* native build + tests on Linux, macOS and Windows
* WebAssembly build + tests via wasmtime
* `parser_upstream` build + cross-impl tests on Linux and macOS
* `--validate-refs` round-trip on every classic reference

A reference that does not round-trip through upstream's parser fails CI;
the `VALIDATE_REFS_SKIP` set in `run_tests.py` is enforced to only ever
shrink.

## Layout

* `mlir_api.h` — the C API surface (handles, op/type/attribute getters,
  builders). Both backends implement this header.
* `mlir_api_impl.c` — native implementation. Self-contained, builds with
  `-nostdlib`, no LLVM dependency.
* `mlir_api_impl_upstream.cpp` — upstream implementation. Wraps
  `mlir::Operation`/`mlir::Type` etc. Compiled into `parser_upstream`.
* `tokenizer.re` / `tokenizer.c` — re2c-generated tokenizer.
  `tokenizer.h` declares the token enum and the tiny tokenizer API.
* `mlir_parser.c` — recursive-descent parser; dispatches per-op to handlers
  in `op_parsers.c`.
* `op_parsers.c` — one parser function per op kind that needs special
  syntax (e.g. `linalg.fill`, `gpu.launch`, `affine.load`,
  `tensor.collapse_shape`).
* `mlir_classic_printer.c` — pretty (typed) form printer. Big switch on
  `MLIR_GetOpType(op)`; ops without a typed-form case fall through to the
  generic-form path.
* `mlir_generic_printer.c` — `"op.name"(operands) : (intypes) -> outtypes`
  printer.
* `mlir_op_names.c` — op enum ↔ string mapping.
* `parser.c` — native CLI: `./parser [--construct] [--parse=K] [--print=K] file`.
* `corec/`, `corec-stdlib/` — pinned submodules; the only sources of
  allocators, strings, formatting, I/O and the stdlib subset.
* `tests/` — input `.mlir` / `.ttir` files, `tests.toml` (per-file expected
  exit code etc.), `tests/reference/` (committed reference output),
  `run_tests.py` (test runner), `tests/mutation_check.py` (test-infra
  mutation sanity check).
* `examples/tinyc/` — an end-to-end compiler example: a tiny C subset
  compiled to a native binary using only `mlir_api.h`. See
  `examples/tinyc/README.md`.

## Tokenizer design

* **Pure tokens, no semantic action.** `tokenizer_next` returns
  `(token_type, first, last)` and nothing else. `first`/`last` are byte
  indices into the source string, not pointers.
* **Maximally informative token kinds.** `TK_PLUS`, `TK_LBRACE`,
  `TK_REGISTER`, `TK_INTEGER`, … so the parser rarely needs to inspect the
  underlying bytes. Numeric and string contents are decoded later, in the
  parser, from `[first, last)`.
* **The tokenizer never fails.** Unrecognised input becomes `TK_ERROR` with
  full location info; the caller decides whether to recover or abort.
* **Reusable shape.** Adapting the tokenizer to a different language is a
  matter of editing `tokenizer.re` (the re2c spec) and the token enum in
  `tokenizer.h`. Run `pixi run tokenizer` to regenerate `tokenizer.c`.

## Parser design

The parser is hand-written, recursive-descent, single-pass, building typed
AST nodes through `mlir_api.h`. Per-op specialisation lives in
`op_parsers.c`:

* Standard MLIR — `arith.*`, `func.*`, `cf.*`, `scf.*`, `memref.*`,
  `tensor.*`, `linalg.*`, `affine.*`, `gpu.*`, `index.*`, `vector.*`.
* Triton IR — `tt.*` ops in `*.ttir` files (treated as unregistered
  generic-form ops where no typed form is needed).

`mlir_parser.c::parse_op_at_top_level` dispatches on the op-name keyword
and routes to the corresponding `parse_*_op` function.

## The classic vs. generic form

Two MLIR textual forms matter to this project:

* **Generic form** — `%r = "op.name"(%a, %b) {attr = ...} : (T1, T2) -> T3`.
  Mechanically printable from `(opname, operands, attrs, result types)` —
  no per-op knowledge required. Always emitted by the generic printer.
* **Classic (typed) form** — `%r = arith.addi %a, %b : i32`,
  `linalg.fill ins(%v : T) outs(%t : T)`, etc. Syntactic sugar that absorbs
  certain attributes (`operandSegmentSizes`, `map`, `reassociation`, region
  bodies, …) into structural positions. Each typed form must be hand-written.

Some upstream ops only verify in classic form because their structural
attributes (e.g. `tensor.collapse_shape`'s `reassociation`,
`affine.load`'s `map`, `linalg.fill`/`linalg.copy`'s `operandSegmentSizes`
and implicit region) cannot be reconstructed from the generic-form bytes
the API normally carries. For those ops we write a typed printer in
`mlir_classic_printer.c` and explicitly skip the absorbed attributes from
the trailing `{...}` block.

## The unified parse/print API

`./parser` and `./parser_upstream` both accept:

```
./parser            [--parse=classic]                  [--print=classic|generic] file.mlir
./parser_upstream   [--parse=classic|upstream] [--print=classic|generic|upstream] file.mlir
./parser --construct                # programmatically build & print a test module
```

The native binary has only the classic parser (the generic and upstream
parse kinds are upstream-only). Both binaries can emit classic and
generic form; only `parser_upstream` can emit upstream pretty form. The
combinations exercised by the test runner live in
`run_tests.py::COMBOS_CLASSIC_PARSER` and `COMBOS_UPSTREAM_PARSER`.
Adding a new `MLIR_PARSE_KIND_*` or `MLIR_PRINT_KIND_*` requires updating
those COMBO arrays — see *Hardening* below.

The naming convention for the dispatch entry points is
`MLIR_<Verb><Subject><Kind>`, e.g. `MLIR_PrintOperationClassic`,
`MLIR_PrintOperationGeneric`, `MLIR_PrintOperationUpstream`. There is no
runtime enum dispatch — each kind is just a function symbol.

## Test infrastructure

`run_tests.py` runs every input under every combo, diffs against the
committed `tests/reference/` output, and reports failures. Guardrails it
enforces:

1. **Hard-fail when a binary is missing.** Combos do not silently skip if
   `parser` or `parser_upstream` is absent.
2. **`tests.toml` covers every input.** Any `tests/*.mlir|*.ttir` not listed
   in `tests.toml` is an error.
3. **`n > 0`.** "All 0 tests passed" is treated as failure.
4. **`--validate-refs`.** Every `*.classic.*` reference must parse cleanly
   through the upstream parser. The `VALIDATE_REFS_SKIP` set is currently
   empty and enforced to only ever shrink.
5. **Mutation sanity check.** `python tests/mutation_check.py` injects a
   small fixed list of one-line bugs across each code path (native classic
   printer, native generic printer, native parser dispatch,
   `--validate-refs` round-trip) and asserts the corresponding suite
   fails. A "still passed" result flags a coverage hole.

`run_tests.py -u` regenerates references; `run_tests.py --validate-refs`
runs the round-trip check; `run_tests.py -s` runs sequentially for clearer
output on failure. CI never invokes `-u`.

## Memory management

All allocation goes through `corec`'s arenas. Lifetime is per-arena: create
once, allocate freely, drop the arena to free everything. There is no
individual `free`.

```c
Arena *arena = arena_create(50 * 1024 * 1024);
MyType *p = arena_alloc(arena, MyType);
MyType *xs = arena_new_array(arena, MyType, count);
// ... use p, xs ...
arena_destroy(arena);   // frees every allocation in one call
```

Vectors are header-only generics: `DEFINE_VECTOR_FOR_TYPE(T, VecT)` then
`VecT_push_back(arena, &v, x)`. The arena is passed in explicitly so the
container never owns memory.

## String handling

The native build does not have a C standard library. Strings are
`corec`'s `string` (`{ const char *str; size_t size; }`) — never assumed
null-terminated.

```c
string s = str_lit("hello");          // compile-time literal
bool eq = str_eq(a, b);               // value equality
string c = str_concat(arena, a, b);   // arena-allocated concat
string f = format(arena, str_lit("Hello {}, {} items"), name, (int64_t)n);
```

`format`'s `{}` placeholder is type-driven — pass an `int64_t` for an
integer, a `string` for a string, etc. The native build's `format`
supports the standard `%d %s %c %x %p %f %.Nf %%` specifiers when invoked
through `printf` from `corec-stdlib`.

A subtle gotcha: `str_lit(LIT)` uses `sizeof(LIT) - 1`, which only works
on **string literals** or fixed `char[]` arrays. Passing a
`const char *kw` produces a length of `sizeof(void *) - 1`. If a keyword
arrives as a `const char *`, copy it into a `char[]` first or convert via
`str_from_cstr_view(p)`.

## Adding a new op

1. Add an enum value to `MLIR_OpType` in `mlir_api.h` (and to the upstream
   side if classification differs).
2. Add the `(string → enum)` row to `mlir_op_names.c`.
3. If the op needs special syntax, write `parse_<op>_op` in `op_parsers.c`
   and route to it from `mlir_parser.c`.
4. If the op has a typed (classic) form, add a case in
   `mlir_classic_printer.c`'s big switch. Skip any attributes the typed
   form absorbs from the trailing `{...}` attr-list block (see the
   `OP_TYPE_AFFINE_LOAD`, `OP_TYPE_TENSOR_COLLAPSE_SHAPE` cases for
   examples).
5. Drop a test input under `tests/`, add it to `tests.toml`, and
   `pixi run -e upstream python run_tests.py -u` to materialise the
   references. Then run without `-u` and `--validate-refs` to confirm.

## Modifying the tokenizer

```bash
$EDITOR tokenizer.re
pixi run tokenizer        # re2c regenerates tokenizer.c (committed to the tree)
pixi run -e macos test_macos
```

`tokenizer.c` is committed so users without re2c installed can still build.

## Entry point

The native binary uses Core C's `int app_main(void)` entry point — there
is no `main`, no C runtime, and no implicit libc dependency. `parser.c`
defines `app_main`; `corec/platform/platform_*.c` is what calls it. The
upstream binary is a normal C++ program with `int main(...)` because it
links real LLVM/MLIR (and therefore libc) anyway.

## License

See [LICENSE](LICENSE).

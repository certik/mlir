# corec bootstrap blockers

This document tracks what tinyC features were added on the
`tinyc-corec-bootstrap` branch in order to compile the
[corec](https://github.com/certik/corec) project, and what blockers remain
that prevent a full bootstrap. It is paired with the corec-side branch
`tinyc-test-macos`, which adds a `test_macos_tinyc` pixi task that
preprocesses corec sources with `clang -E -P -nostdinc -fno-builtin`
and then feeds each `.i` file through the `tinyc` driver.

## Features added to tinyC for corec

These are all front-end / lexer-only changes; no major emit refactor.
All 80 tinyC tests (2 skipped on darwin) continue to pass.

- **Lexer**: keywords `short`, `_Bool`, `bool`, `double` (alias of
  `float`), `__builtin_va_list` (alias of `va_list`).
- **Type-modifier parsing**: `parse_base_type` and `parse_sig_type`
  now accept any combination of `signed`/`unsigned`/`short`/`long`/
  `int`/`char`/`_Bool`/`bool`. `long` (or `long long`) maps to
  `TY_I64`; everything else maps to `TY_I32`.
  A trailing `*` on a `char`-modified type produces `TY_PTR_CHAR`.
- **Anonymous struct typedef**: `typedef struct { ... } Alias;` is
  parsed by allowing `parse_struct_def` to omit the tag and patching
  `sd->name = alias` at the typedef call site.
- **`typedef struct Tag Alias;` (opaque)**: registers an empty
  `StructDef` for `Tag` so that pointer-only uses (`Alias *`) can be
  resolved by the emitter's `find_struct`.
- **`typedef struct Tag { ... } Alias;`**: parses the body and registers
  both the struct and the typedef.
- **`typedef enum [Tag] { ... } Alias;`**: extracted a `parse_enum_body`
  helper from `parse_enum_decl_top` and reuses it from a new typedef-enum
  branch. `Alias` becomes a typedef for `int`; enumerators are
  registered as int constants.
- **`typedef __builtin_va_list ...;`**: top-level no-op so that
  `<stdarg.h>`-style headers parse.
- **`struct Tag;` forward declaration**: registers an empty `StructDef`
  placeholder for the tag.
- **Struct field as a typedef name**: struct field parsers now resolve
  typedef names. A trailing `*` produces `TY_PTR_STRUCT` /
  `TY_PTR_VOID` / `TY_PTR_I32` (i64 pointers are bucketed to
  `TY_PTR_I32`).
- **Brace-less control-flow body**: `parse_block` now accepts a single
  statement where a braced block is expected, so `if (cond) stmt;`,
  `else stmt;`, etc. are accepted directly.
- **`va_list` in parameter / typedef position**: parses cleanly and is
  lowered by the emitter as an opaque pointer-sized value
  (`scalar_mlir_type` returns `e->ptr` for `TY_VA_LIST`).
- **`TY_I64` struct fields**: registered in `init_struct_types` so a
  64-bit field has a valid backing MLIR type.

New tinyC tests covering the language additions:
- `tests/braceless_if.tc`
- `tests/typedef_struct_anon.tc`
- `tests/typedef_enum.tc`

## Remaining blockers (deferred)

After the changes above, `pixi run -e macos test_macos_tinyc` parses
`platform/test_base_only.c` cleanly but stops in the emitter on the
following issues. Each of these requires a non-trivial emit-side
refactor (well beyond the agreed budget for this bootstrap PR), so
they are documented here rather than fixed.

| # | Symptom (preprocessed line) | Blocker | Rough scope |
|---|---|---|---|
| 1 | `field size is not a scalar lvalue` | Load/store of `TY_I64` struct fields. The emit-time field-access code path special-cases `TY_I32` / `TY_F32` / pointer fields, but i64 fields fail when assigned-from / read-as a scalar expression. Needs the same `TY_I64` plumbing applied at the ~5 field-access sites in `emit.c` (search for `field size is not a scalar lvalue`). | ~80 LOC across `emit.c` |
| 2 | `variadic function format_explicit cannot return a struct` | `string` (a 16-byte `{char*; uint64}` struct) is returned by value from a variadic function. tinyC currently forbids struct return from variadic functions because the arg-pack ABI for struct returns is unimplemented. | ~150 LOC: variadic-frame restructuring |
| 3 | `unsupported type in function signature` (was: `va_list ap` parameter) | **Already fixed** — `TY_VA_LIST` is now allowed in `slot_resolve` and `scalar_mlir_type`. Listed here for completeness. | done |
| 4 | Untested but expected: `va_arg(ap, T)` calls in actual function bodies. tinyC's `va_arg` is parser-only at the moment; lowering it requires platform-specific frame walks. | ~200 LOC in `emit.c` plus a runtime helper |
| 5 | Untested but expected: function-pointer struct fields (`vtable`-style). tinyC parses `typedef R (*F)(...);` and uses these as variables, but field-of-fnptr requires `TY_FNPTR` to be wired into the field-access paths the same way `TY_PTR_*` is. | ~100 LOC in `emit.c` |
| 6 | Untested but expected: aggregate initializers (`Point p = {1,2};`, `int xs[3] = {1,2,3};`). tinyC currently rejects array initializers and only supports compound literals as r-values. | ~250 LOC across parse + emit |
| 7 | Untested but expected: explicit signed/unsigned 8/16-bit integer types as values (not just typedefs that fold to `int`). Treating `uint8_t` as `int` works for most arithmetic but breaks for sizeof-based bookkeeping, struct layout, and pointer arithmetic over `uint8_t*` buffers. | ~400 LOC across the type system |
| 8 | Untested but expected: GNU-style `__attribute__((...))` and `__inline__` attribute syntax surfaced by `clang -E` of the macOS system headers — currently corec avoids them by `-nostdinc`, but a non-`-nostdinc` build would surface this immediately. | parser-only, ~50 LOC |

The total estimated cost to make corec fully compile with tinyC is on
the order of ~1000–1500 LOC of emit-side changes, dominated by items
1–2 and 6. That exceeds the agreed scope for this PR, so the PR ships
the parser/lexer additions plus the corec-side `test_macos_tinyc`
plumbing and stops short of full bootstrap.

## How to reproduce

```sh
# tinyC tests (this branch):
cd mlir
pixi run -e upstream test_tinyc_upstream

# corec attempt (corec branch tinyc-test-macos, requires this branch's tinyc):
cd corec
bash scripts/build_macos_tinyc.sh   # or: pixi run -e macos test_macos_tinyc
```

The corec build will fail with the diagnostics in the table above.
The first two items are the immediate blockers for `test_base_only.c`.

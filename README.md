# MLIR

Faster and simpler MLIR infrastructure.

This project is built on top of [corec](https://github.com/certik/corec) and
[corec-stdlib](https://github.com/certik/corec-stdlib), which are included as
git submodules. Builds are driven by [pixi](https://pixi.sh) and target
Linux, macOS, Windows and WebAssembly (WASI) using a `-nostdlib -nostdinc
-fno-builtin` toolchain (no system C runtime).

## Getting started

Clone the repository with submodules:

```bash
git clone --recursive https://github.com/certik/mlir.git
cd mlir
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

## Build & test

Pick the environment for your host and run the corresponding test task:

```bash
pixi run -e macos   test_macos       # macOS (arm64)
pixi run -e linux   test_linux       # Linux (x86_64)
pixi run -e windows test_windows     # Windows (MSVC; run from a vcvars64 shell)
pixi run -e wasm    test_wasm        # WebAssembly via wasmtime
```

Each `test_*` task builds the parser binary (named `parser`, or `parser.exe`
/ `parser.wasm`) and runs `./parser --construct` followed by the diff-test
suite via `python run_tests.py -s`.

To just build without running the tests, use `build_<platform>` in place of
`test_<platform>`.

To regenerate test references after intentional output changes:

```bash
pixi run -e macos python run_tests.py -s -u
```

# Tokenizer Design

* Just return token type, first, last. Probably first/last should be integer
  indices into the string, not pointers.
* The tokenizer is thus as efficient as it can be, since we have to return
  location information and token type anyway and we do not return nor do
  anything else.
* The API surface to use the tokenizer is also minimal, and the tokenizer code
  can be easily reused / adapted for any other tokenizer easily, since just the
  list of tokens enum and the re2c code needs to be updated.
* The tokenizer returns an error token on errors, so the caller can decide how
  to handle it (e.g., reporting an error and optionally continue). The error
  token still has location information, so the caller can return an error like
  `"Token '" + t + "' is not recognized"`. Consequently the tokenizer cannot
  fail and does not report any errors or warnings, the caller does everything.
* There is one issue that if the list of token types are coming from Bison, we
  need to append more token types to the list that we do not want Bison to
  handle, such as the error token(s), or various temporary tokens to handle
  warnings, none of which will be passed on to the parser.
* If we don't want that, we can just use the token types from Bison, but return
  other flags that the caller can check, like an error flag, a warning flag for
  some tokens, etc.
* There can be a thin layer between the tokenizer and the parser that ignores
  some tokens (such as white space) or returns some warnings / errors.
* One can skip whitespace (both first/last needed), but can also return it as a
  token (just `last` needed, but not a huge overhead to return `first` also,
  making the parser independent if we skip characters or not in the tokenizer).
* We try to return as much information as we can as part of the token type,
  such as `TK_PLUS`, `TK_MINUS`, `TK_LEFT_PAREN`, etc., that way we do not need
  to consult the string at all for those. Only for tokens like string or number
  we need to actually handle those in some way in the parser, although even
  those we can postpone converting them until later in the semantic phase.
* In the parser's actions we can extract the string using first/last and
  convert / further process as needed based on the syntax information from the
  parser or possibly even convert later based on semantic information.

# Parser Design

The tokenizer seems context dependent. In that case, just create multiple
tokenizers.


* Just return token type, first, last. Probably first/last should be integer
  indices into the string, not pointers.
* The tokenizer is thus as efficient as it can be, since we have to return
  location information and token type anyway and we do not return nor do
  anything else.
* The API surface to use the tokenizer is also minimal, and the tokenizer code
  can be easily reused / adapted for any other tokenizer easily, since just the
  list of tokens enum and the re2c code needs to be updated.
* The tokenizer returns an error token on errors, so the caller can decide how
  to handle it (e.g., reporting an error and optionally continue). The error
  token still has location information, so the caller can return an error like
  `"Token '" + t + "' is not recognized"`. Consequently the tokenizer cannot
  fail and does not report any errors or warnings, the caller does everything.
* There is one issue that if the list of token types are coming from Bison, we
  need to append more token types to the list that we do not want Bison to
  handle, such as the error token(s), or various temporary tokens to handle
  warnings, none of which will be passed on to the parser.
* If we don't want that, we can just use the token types from Bison, but return
  other flags that the caller can check, like an error flag, a warning flag for
  some tokens, etc.
* There can be a thin layer between the tokenizer and the parser that ignores
  some tokens (such as white space) or returns some warnings / errors.
* One can skip whitespace (both first/last needed), but can also return it as a
  token (just `last` needed, but not a huge overhead to return `first` also,
  making the parser independent if we skip characters or not in the tokenizer).
* We try to return as much information as we can as part of the token type,
  such as `TK_PLUS`, `TK_MINUS`, `TK_LEFT_PAREN`, etc., that way we do not need
  to consult the string at all for those. Only for tokens like string or number
  we need to actually handle those in some way in the parser, although even
  those we can postpone converting them until later in the semantic phase.
* In the parser's actions we can extract the string using first/last and
  convert / further process as needed based on the syntax information from the
  parser or possibly even convert later based on semantic information.

# Parser Design

The tokenizer seems context dependent. In that case, just create multiple
tokenizers.

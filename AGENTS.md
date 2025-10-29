# AGENTS.md

This file provides guidance to Agents when working with code in this repository.

## Project Overview

This is a custom MLIR (Multi-Level Intermediate Representation) infrastructure implementation written in C. It's designed to be faster and simpler than the standard LLVM MLIR implementation, focusing on parsing, tokenizing, and printing MLIR code.

### Key Components

- **Tokenizer**: Uses re2c for high-performance lexical analysis (`tokenizer.re` -> `tokenizer.c`)
- **Parser**: Recursive descent parser that builds AST from tokens (`mlir_parser.c/h`)
- **Main Parser**: Command-line interface for parsing MLIR files (`parser.c`)
- **Base Utilities**: Arena allocator, string handling, I/O, formatting (`base/` directory)
- **Testing**: Unit tests and diff tests (`run_tests`, `tests/`)

### Architecture

The codebase follows a modular C architecture with clear separation of concerns:

1. **Memory Management**: Arena-based allocation for fast, leak-free memory management
2. **Tokenization**: re2c-generated tokenizer for efficient lexical analysis
3. **Parsing**: Hand-written recursive descent parser building typed AST nodes
4. **Representation**: C structs representing MLIR concepts (Operations, Blocks, Regions, Types, Attributes)
5. **Printing**: Generic printer that can output MLIR in standard format

## Build System

### Prerequisites
- `clang` compiler
- `re2c` lexer generator (for tokenizer generation)
- AddressSanitizer support (used by default for debugging)

### Build Commands

**Primary build script:**
```bash
./build.sh
```

This script:
1. Compiles and runs unit tests (`run_tests`)
2. Generates tokenizer from `tokenizer.re` using re2c
3. Builds main parser binary
4. Runs parser in construction mode and file parsing mode

**Windows build:**
```bash
./msvc_build.sh  # Uses cl.exe with MSVC
```

**Manual compilation:**
```bash
# Generate tokenizer (required before compilation)
re2c --no-generation-date -b tokenizer.re -o tokenizer.c

# Build with AddressSanitizer
CFLAGS="-fsanitize=address -g -Wall -ferror-limit=1"
clang $CFLAGS -I. -o parser parser.c tokenizer.c mlir_parser.c base/arena.c base/string.c base/format.c base/io.c

# Build tests
clang $CFLAGS -I. -o run_tests tests/run_tests.c base/arena.c base/string.c base/format.c base/io.c
```

## Testing

After every change ensure we can build the code and run all tests:
```
./build.sh
uv run run_tests.py -s
```
The `-s` option runs tests sequentially which is useful for better
understanding the output if any test fails.

To update reference results, do:
```
uv run run_tests.py -u
```
This is needed when our printer or other internals change and the changes are
good and we want to keep them.

### Test Structure
- `tests/run_tests.c`: Unit tests for base utilities (arena, string, format, I/O)
- `tests/tests.toml`: Test configuration with expected outputs
- `run_tests.py`: Python test runner with comparison logic

## Usage Modes

### Parser Binary
The `parser` binary supports two main modes:

**Construction Mode** (for testing):
```bash
./parser --construct
```
Creates a test MLIR module programmatically and prints it. Used for validating the printer output format.

**File Parsing Mode**:
```bash
./parser tests/simple.mlir
./parser tests/add_kernel.ttir
```
Parses MLIR files and prints the parsed representation.

## Key Files and Directories

### Core Implementation
- `parser.c`: Main CLI application and demo/test code
- `mlir_parser.h/c`: MLIR parsing logic and AST data structures
- `tokenizer.re`: re2c tokenizer specification (generates `tokenizer.c`)
- `tokenizer.h`: Token type definitions and tokenizer API

### Base Utilities (`base/`)
- `arena.h/c`: Memory arena allocator for fast allocation/deallocation
- `string.h/c`: Custom string type and string operations
- `format.h/c`: Printf-style string formatting with custom syntax
- `io.h/c`: File I/O operations
- `vector.h`: Generic vector implementation (header-only)
- `hashtable.h`: Hash table implementation (header-only)

### Test Files
- `tests/simple.mlir`: Basic MLIR with control flow
- `tests/*.ttir`: Triton TTIR (Triton IR) files
- `tests/*.mlir`: Standard MLIR files

## Development Workflow

### Modifying the Tokenizer
1. Edit `tokenizer.re` with re2c syntax
2. Regenerate: `re2c --no-generation-date -b tokenizer.re -o tokenizer.c`
3. Rebuild and test

### Adding New Operation Types
1. Add enum value to `MLIR_OpType` in `mlir_parser.h`
2. Update `op_type_to_string()` function in `parser.c`
3. Add parsing logic in `mlir_parser.c` if needed
4. Add test cases

### Memory Management
- Always use arena allocation: `arena_alloc(arena, Type)` or `arena_alloc_array(arena, Type, count)`
- No need for individual `free()` calls - arena cleanup handles everything
- Create arena with sufficient size: `arena_create(50*1024*1024)` for large workloads

## Code Patterns and Conventions

### Error Handling
- Functions return boolean for success/failure where appropriate
- Use `assert()` for invariant checking in debug builds
- Print errors to stderr and exit with non-zero code on fatal errors

### String Handling
- Use custom `string` type (not null-terminated char*)
- String literals: `str_lit("hello")`
- String operations: `str_concat()`, `str_eq()`, `str_substr()`
- Format strings: `format(arena, str_lit("Hello {}"), name)`

### Vector Usage
- Define vectors with `DEFINE_VECTOR_FOR_TYPE(Type, VectorName)`
- Use generated functions like `VectorName_init()`, `VectorName_append()`

### Parsing Extensions
The parser handles both standard MLIR and Triton TTIR extensions:
- Standard MLIR: `.mlir` files with `arith.*`, `func.*`, `cf.*` operations
- Triton IR: `.ttir` files with `tt.*` operations and specialized syntax

## Common Issues and Solutions

### Tokenizer Changes Not Applied
**Problem**: Modified `tokenizer.re` but changes don't appear
**Solution**: Run `re2c --no-generation-date -b tokenizer.re -o tokenizer.c` to regenerate

### AddressSanitizer Errors
**Problem**: Memory errors detected by ASan
**Solution**: 
- Check arena allocation bounds
- Ensure proper string/array bounds
- Verify pointer validity before dereferencing

### Parser Test Failures
**Problem**: Parser produces different output than expected
**Solution**:
- Compare with reference files in `tests/reference/`
- Use `--construct` mode to validate printer logic
- Check tokenizer output first with debug flag in `tokenizer_print_all_tokens()`

### Build Failures on Different Platforms
**Problem**: Compilation issues
**Solution**: 
- Unix/Linux/macOS: Use `build.sh` with clang
- Windows: Use `msvc_build.sh` with Visual Studio tools
- Adjust CFLAGS if AddressSanitizer not available

## Performance Considerations

- **Arena allocation**: Very fast allocation, batch deallocation
- **re2c tokenizer**: Highly optimized lexer with minimal overhead
- **Single-pass parsing**: Parser builds AST in one pass through tokens
- **Minimal copying**: String operations work with views/slices where possible

## Future Extensions

The codebase is designed for easy extension:
- **New dialects**: Add operation types and parsing rules
- **Optimization passes**: AST is mutable for transformations
- **Different output formats**: Extend printer for JSON, custom formats
- **Language bindings**: C API makes bindings straightforward

## External Dependencies

- **re2c**: Required for tokenizer generation
- **clang**: Preferred compiler (GCC should work too)
- **Python 3.11+**: For test runner (`run_tests.py`)
- **toml**: Python library for test configuration parsing

The implementation minimizes external dependencies and uses custom base utilities instead of standard libraries where beneficial for performance or control.

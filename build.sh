#!/bin/bash

set -ex

# Use AddressSanitizer on macOS, but not on Linux due to WSL issues
if [[ "$OSTYPE" == "darwin"* ]]; then
    CFLAGS="-fsanitize=address -g -Wall -ferror-limit=1"
else
    CFLAGS="-g -Wall -ferror-limit=1"
fi

clang $CFLAGS -I. -o run_tests tests/run_tests.c base/arena.c base/string.c base/format.c base/io.c
#./run_tests

re2c --no-generation-date -b tokenizer.re -o tokenizer.c
clang $CFLAGS -I. -o parser parser.c tokenizer.c mlir_parser.c mlir_generic_printer.c mlir_classic_printer.c op_parsers.c mlir_api_impl.c base/arena.c base/string.c base/format.c base/io.c
./parser
./parser --construct

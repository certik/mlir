#!/bin/bash

set -ex

CFLAGS="-fsanitize=address -g -Wall -ferror-limit=1"

clang $CFLAGS -I. -o run_tests tests/run_tests.c base/arena.c base/string.c base/format.c base/io.c
./run_tests

re2c --no-generation-date -b tokenizer.re -o tokenizer.c
clang $CFLAGS -I. -o parser parser.c tokenizer.c mlir_parser.c base/arena.c base/string.c base/format.c base/io.c
./parser
./parser --construct
./run_mlir_tests.sh

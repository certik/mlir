#!/bin/bash

set -ex

CFLAGS="-fsanitize=address -g -Wall -ferror-limit=1"

clang $CFLAGS -I. -o test_format tests/test_format.c base/arena.c base/string.c base/format.c
./test_format

re2c -b tokenizer.re -o tokenizer.c
clang $CFLAGS -I. -o parser parser.c tokenizer.c base/arena.c base/io.c mlir_parser.c
./parser

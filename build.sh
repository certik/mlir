#!/bin/bash

set -ex

CFLAGS="-fsanitize=address -g -Wall -ferror-limit=1"

re2c -b tokenizer.re -o tokenizer.c
clang $CFLAGS -c tokenizer.c -o tokenizer.o
clang $CFLAGS -c arena.c -o arena.o
clang $CFLAGS -c mlir_parser.c -o mlir_parser.o
clang++ $CFLAGS -std=c++20 -c parser.cpp -o parser.o
clang++ $CFLAGS -std=c++20 -o parser parser.o tokenizer.o arena.o
./parser

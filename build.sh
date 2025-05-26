#!/bin/bash

set -ex

re2c -b tokenizer.re -o tokenizer.c
clang -g -Wall -ferror-limit=1 -c tokenizer.c -o tokenizer.o
clang++ -g -Wall -std=c++20 -ferror-limit=1 -c parser.cpp -o parser.o
clang++ -g -Wall -std=c++20 -o parser parser.o tokenizer.o
./parser

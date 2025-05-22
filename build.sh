#!/bin/bash

set -ex

re2c -b tokenizer.re -o tokenizer.cpp
clang++ -g -Wall -std=c++20 parser.cpp tokenizer.cpp -o parser

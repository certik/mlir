#!/bin/bash

set -ex

clang++ -g -Wall -std=c++17 parser.cpp -o parser

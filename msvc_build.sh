set -ex

cl /std:c11 /Zc:preprocessor /I. /Fe:run_tests.exe tests/run_tests.c base/arena.c base/string.c base/format.c base/io.c
./run_tests

cl /std:c11 /Zc:preprocessor /I. /Fe:parser.exe parser.c tokenizer.c mlir_parser.c base/arena.c base/string.c base/format.c base/io.c base/vector.c
./parser

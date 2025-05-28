set -ex

cl /Zc:preprocessor /I. /Fe:test_format.exe tests/test_format.c base/arena.c base/string.c base/format.c
./test_format

#!/usr/bin/env bash
# Run the MLIR test suite using parser_upstream in place of parser.
# parser is restored on exit. Many tests are expected to fail because the
# upstream MLIR backend can't fully represent everything our native impl
# does (comments, register names, ref locations, ...).

set -u
cd "$(dirname "$0")/.."

cleanup() {
    rc=$?
    rm -f parser
    if [ -e parser.native.bak ]; then
        mv parser.native.bak parser
    fi
    exit "$rc"
}
trap cleanup EXIT INT TERM

if [ -e parser ]; then
    mv parser parser.native.bak
fi
cp parser_upstream parser

set +e
python run_tests.py -s

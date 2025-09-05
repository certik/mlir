#!/bin/bash

# Check roundtripping

set -ex

TEST=add_kernel
./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

TEST=conv2d
./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

TEST=matmul1
./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

TEST=triton_mm
./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

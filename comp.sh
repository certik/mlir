#!/bin/bash

# Check roundtripping

set -ex


TEST=triton_mm
./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

TEST=matmul1
./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

TEST=effect
./parser --classic tests/${TEST}.mlir > ${TEST}2.mlir
diff -Naur tests/${TEST}.mlir ${TEST}2.mlir

TEST=sumrow
./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

TEST=conv2d
./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

TEST=chunked_cross_entropy_forward
./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

TEST=b
./parser --classic tests/${TEST}.mlir > ${TEST}2.mlir
diff -Naur tests/${TEST}.mlir ${TEST}2.mlir

TEST=c
./parser --classic tests/${TEST}.mlir > ${TEST}2.mlir
diff -Naur tests/${TEST}.mlir ${TEST}2.mlir

TEST=add_kernel
./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

# Hand modified:
#TEST=d
#./parser --classic tests/${TEST}.mlir > ${TEST}2.mlir
#diff -Naur tests/${TEST}.mlir ${TEST}2.mlir

# Hand modified:
#TEST=add1
#./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
#diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

# Weird indenting
#TEST=t3
#./parser --classic tests/${TEST}.mlir > ${TEST}2.mlir
#diff -Naur tests/${TEST}.mlir ${TEST}2.mlir

# Weird indenting
#TEST=t2
#./parser --classic tests/${TEST}.mlir > ${TEST}2.mlir
#diff -Naur tests/${TEST}.mlir ${TEST}2.mlir

TEST=simple
./parser --classic tests/${TEST}.mlir > ${TEST}2.mlir
diff -Naur tests/${TEST}.mlir ${TEST}2.mlir

# Weird formatting
#TEST=t1
#./parser --classic tests/${TEST}.mlir > ${TEST}2.mlir
#diff -Naur tests/${TEST}.mlir ${TEST}2.mlir

# Generic printing (not classic)
#TEST=a
#./parser --classic tests/${TEST}.mlir > ${TEST}2.mlir
#diff -Naur tests/${TEST}.mlir ${TEST}2.mlir

#!/bin/bash

# Check roundtripping

set -ex


TEST=triton_mm
./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

TEST=matmul1
./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

#effect.mlir
#sumrow.ttir

TEST=conv2d
./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

#chunked_cross_entropy_forward.ttir
#b.mlir
#c.mlir

TEST=add_kernel
./parser --classic tests/${TEST}.ttir > ${TEST}2.ttir
diff -Naur tests/${TEST}.ttir ${TEST}2.ttir

#d.mlir
#add1.ttir
#t3.mlir
#t2.mlir
#simple.mlir
#t1.mlir
#a.mlir




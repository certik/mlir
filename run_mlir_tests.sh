#!/bin/bash

set -ex

./parser test_mlir/a.mlir
./parser test_mlir/b.mlir
./parser test_mlir/c.mlir
./parser test_mlir/d.mlir
./parser test_mlir/conv2d.ttir
./parser test_mlir/chunked_cross_entropy_forward.ttir
./parser test_mlir/sumrow.ttir
./parser test_mlir/triton_mm.ttir
./parser test_mlir/add1.ttir
./parser test_mlir/matmul1.ttir
./parser test_mlir/effect.mlir
./parser test_mlir/simple.mlir
./parser test_mlir/t1.mlir
./parser test_mlir/t2.mlir
./parser test_mlir/t3.mlir
./parser test_mlir/add_kernel.ttir

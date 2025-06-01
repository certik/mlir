#!/bin/bash

set -ex

./parser test_mlir/a.mlir
./parser test_mlir/b.mlir
./parser test_mlir/c.mlir
./parser test_mlir/d.mlir
./parser test_mlir/conv2d.ttir

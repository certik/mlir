// Op facts shared by the LLVM-dialect optimisation passes (mlir_llvm_dce.c,
// mlir_llvm_load_cse.c, mlir_llvm_arith_gvn.c), so they cannot drift apart.
//
// Header-only `static inline` so no extra translation unit has to be wired
// into every build script (see mlir_codegen_buf.h).

#pragma once

#include <stdbool.h>

#include "mlir_api.h"

// Side-effect free: no memory access, no control flow, no call. Such an op
// never clobbers memory, and it can be erased once its result is unused.
// Loads, stores, calls, terminators and anything unrecognised are not pure.
static inline bool llvm_op_is_pure(MLIR_OpType type) {
    switch (type) {
    case OP_TYPE_LLVM_ADD: case OP_TYPE_LLVM_SUB: case OP_TYPE_LLVM_MUL:
    case OP_TYPE_LLVM_UDIV: case OP_TYPE_LLVM_SDIV:
    case OP_TYPE_LLVM_UREM: case OP_TYPE_LLVM_SREM:
    case OP_TYPE_LLVM_AND: case OP_TYPE_LLVM_OR: case OP_TYPE_LLVM_XOR:
    case OP_TYPE_LLVM_SHL: case OP_TYPE_LLVM_LSHR: case OP_TYPE_LLVM_ASHR:
    case OP_TYPE_LLVM_ICMP:
    case OP_TYPE_LLVM_ZEXT: case OP_TYPE_LLVM_SEXT: case OP_TYPE_LLVM_TRUNC:
    case OP_TYPE_LLVM_PTRTOINT: case OP_TYPE_LLVM_MLIR_CONSTANT:
    case OP_TYPE_LLVM_MLIR_ADDRESSOF:
    case OP_TYPE_LLVM_INTTOPTR: case OP_TYPE_LLVM_BITCAST:
    case OP_TYPE_LLVM_GEP: case OP_TYPE_LLVM_SELECT:
    case OP_TYPE_LLVM_FADD: case OP_TYPE_LLVM_FSUB:
    case OP_TYPE_LLVM_FMUL: case OP_TYPE_LLVM_FDIV:
    case OP_TYPE_LLVM_FNEG: case OP_TYPE_LLVM_FCMP:
    case OP_TYPE_LLVM_FPEXT: case OP_TYPE_LLVM_FPTRUNC:
    case OP_TYPE_LLVM_SITOFP: case OP_TYPE_LLVM_UITOFP:
    case OP_TYPE_LLVM_FPTOSI: case OP_TYPE_LLVM_FPTOUI:
        return true;
    default:
        return false;
    }
}

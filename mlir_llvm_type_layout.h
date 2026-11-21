// Size / alignment / struct layout of LLVM-dialect types, shared by the
// backends (mlir_llvm_to_aarch64.c, mlir_llvm_to_x64_emit.inc,
// mlir_llvm_to_wasmssa.c). The layout is the natural one of the LLVM data
// layouts they target: every scalar is aligned to its size, a struct field to
// its own alignment, with trailing padding to the struct's largest alignment.
// The targets differ only in the pointer size, which the caller passes.
//
// Header-only `static inline` so no extra translation unit has to be wired
// into every build script (see mlir_codegen_buf.h).

#pragma once

#include <stddef.h>
#include <string.h>

#include "mlir_api.h"

static inline unsigned llvm_type_size(MLIR_Context *ctx, MLIR_TypeHandle ty,
                                      unsigned ptr_bytes);

// Round x up to a multiple of `a` (a power of two).
static inline unsigned llvm_align_up(unsigned x, unsigned a) {
    return (x + a - 1) & ~(a - 1);
}

static inline unsigned llvm_type_align(MLIR_Context *ctx, MLIR_TypeHandle ty,
                                       unsigned ptr_bytes) {
    if (MLIR_IsTypeLLVMArray(ty))
        return llvm_type_align(ctx, MLIR_GetTypeLLVMArrayElement(ty), ptr_bytes);
    if (MLIR_IsTypeLLVMStruct(ty)) {
        size_t nf = MLIR_GetTypeLLVMStructNumFields(ty);
        unsigned ma = 1;
        for (size_t i = 0; i < nf; i++) {
            unsigned fa = llvm_type_align(ctx, MLIR_GetTypeLLVMStructField(ty, i), ptr_bytes);
            if (fa > ma) ma = fa;
        }
        return ma;
    }
    unsigned sz = llvm_type_size(ctx, ty, ptr_bytes);
    return sz ? sz : 1;
}

// Byte offset of field `fld` within the LLVM struct `sty`.
static inline unsigned llvm_struct_field_offset(MLIR_Context *ctx, MLIR_TypeHandle sty,
                                                size_t fld, unsigned ptr_bytes) {
    unsigned off = 0;
    for (size_t i = 0; i <= fld; i++) {
        MLIR_TypeHandle ft = MLIR_GetTypeLLVMStructField(sty, i);
        off = llvm_align_up(off, llvm_type_align(ctx, ft, ptr_bytes));
        if (i == fld) return off;
        off += llvm_type_size(ctx, ft, ptr_bytes);
    }
    return off;
}

// Size in bytes, or 0 for a type that has no size (e.g. void, or an
// aggregate containing one).
static inline unsigned llvm_type_size(MLIR_Context *ctx, MLIR_TypeHandle ty,
                                      unsigned ptr_bytes) {
    // Aggregates first: their size comes from the element / field types, and
    // MLIR_GetTypeString() formats (and allocates) the whole nested type on
    // every call, which was a measurable share of compile time and memory.
    if (MLIR_IsTypeLLVMArray(ty)) {
        unsigned esz = llvm_type_size(ctx, MLIR_GetTypeLLVMArrayElement(ty), ptr_bytes);
        if (esz == 0) return 0;
        return esz * (unsigned)MLIR_GetTypeLLVMArrayNumElements(ty);
    }
    if (MLIR_IsTypeLLVMStruct(ty)) {
        size_t nf = MLIR_GetTypeLLVMStructNumFields(ty);
        unsigned off = 0, max_align = 1;
        for (size_t i = 0; i < nf; i++) {
            MLIR_TypeHandle ft = MLIR_GetTypeLLVMStructField(ty, i);
            unsigned fsz = llvm_type_size(ctx, ft, ptr_bytes);
            unsigned fal = llvm_type_align(ctx, ft, ptr_bytes);
            if (fsz == 0 || fal == 0) return 0;
            off = llvm_align_up(off, fal) + fsz;
            if (fal > max_align) max_align = fal;
        }
        return llvm_align_up(off, max_align);
    }
    string s = MLIR_GetTypeString(ctx, ty);
    if (s.size >= 9 && memcmp(s.str, "!llvm.ptr", 9) == 0) return ptr_bytes;
    if (s.size == 3 && memcmp(s.str, "ptr", 3) == 0) return ptr_bytes;
    if (s.size == 3 && memcmp(s.str, "f32", 3) == 0) return 4;
    if (s.size == 3 && memcmp(s.str, "f64", 3) == 0) return 8;
    if (s.size > 1 && s.str[0] == 'i') {
        int w = 0;
        for (size_t i = 1; i < s.size; i++) {
            if (s.str[i] >= '0' && s.str[i] <= '9') w = w * 10 + (s.str[i] - '0');
            else { w = -1; break; }
        }
        if (w == 1 || w == 8) return 1;
        if (w == 16) return 2;
        if (w == 32) return 4;
        if (w == 64) return 8;
    }
    return 0;
}

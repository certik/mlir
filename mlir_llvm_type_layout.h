// Size / alignment / struct layout of LLVM-dialect types, and (declared
// here, defined in mlir_llvm_global_init.c) the memory image of a global's
// constant initializer, shared by the backends
// (mlir_llvm_to_aarch64.c, mlir_llvm_to_x64_emit.inc, mlir_llvm_to_wasmssa.c,
// mlir_translate_to_llvm_ir.c). The layout is the natural one of the LLVM data
// layouts they target: every scalar is aligned to its size, a struct field to
// its own alignment, with trailing padding to the struct's largest alignment.
// The targets differ only in the pointer size, which the caller passes.
//
// The layout functions are small `static inline` ones; the initializer
// helpers are not inline, because tinyC emits every static inline body into
// each translation unit that includes the header.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
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
    if (MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_LLVM_ARRAY)
        return llvm_type_align(ctx, MLIR_GetTypeElement(ctx, ty), ptr_bytes);
    if (MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_LLVM_STRUCT) {
        size_t nf = MLIR_GetTypeLLVMStructNumFields(ctx, ty);
        unsigned ma = 1;
        for (size_t i = 0; i < nf; i++) {
            unsigned fa = llvm_type_align(ctx, MLIR_GetTypeLLVMStructField(ctx, ty, i), ptr_bytes);
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
        MLIR_TypeHandle ft = MLIR_GetTypeLLVMStructField(ctx, sty, i);
        off = llvm_align_up(off, llvm_type_align(ctx, ft, ptr_bytes));
        if (i == fld) return off;
        off += llvm_type_size(ctx, ft, ptr_bytes);
    }
    return off;
}

// True for a valid type of size 0, unlike a type with no size: `[0 x T]`
// with a sized T, an array of such, or a struct of only such fields (or
// none, a GNU C empty struct).
static inline bool llvm_type_is_empty(MLIR_Context *ctx, MLIR_TypeHandle ty,
                                      unsigned ptr_bytes) {
    if (MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_LLVM_ARRAY) {
        MLIR_TypeHandle et = MLIR_GetTypeElement(ctx, ty);
        if (llvm_type_size(ctx, et, ptr_bytes) != 0)
            return MLIR_GetTypeNumElements(ctx, ty) == 0;
        return llvm_type_is_empty(ctx, et, ptr_bytes);
    }
    if (MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_LLVM_STRUCT) {
        size_t nf = MLIR_GetTypeLLVMStructNumFields(ctx, ty);
        if (nf == 0) {
            // An opaque struct (no body, `!llvm.struct<"S", opaque>`) has
            // no size at all.
            string s = MLIR_GetTypeString(ctx, ty);
            if (s.size >= 9 && memcmp(s.str + s.size - 9, ", opaque>", 9) == 0)
                return false;
        }
        for (size_t i = 0; i < nf; i++) {
            if (!llvm_type_is_empty(ctx, MLIR_GetTypeLLVMStructField(ctx, ty, i), ptr_bytes))
                return false;
        }
        return true;
    }
    return false;
}

// Size in bytes, or 0 for a type that has no size (e.g. void, or an
// aggregate containing one).
static inline unsigned llvm_type_size(MLIR_Context *ctx, MLIR_TypeHandle ty,
                                      unsigned ptr_bytes) {
    // Aggregates first: their size comes from the element / field types, and
    // MLIR_GetTypeString() formats (and allocates) the whole nested type on
    // every call, which was a measurable share of compile time and memory.
    if (MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_LLVM_ARRAY) {
        unsigned esz = llvm_type_size(ctx, MLIR_GetTypeElement(ctx, ty), ptr_bytes);
        if (esz == 0) return 0;
        return esz * (unsigned)MLIR_GetTypeNumElements(ctx, ty);
    }
    if (MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_LLVM_STRUCT) {
        size_t nf = MLIR_GetTypeLLVMStructNumFields(ctx, ty);
        unsigned off = 0, max_align = 1;
        for (size_t i = 0; i < nf; i++) {
            MLIR_TypeHandle ft = MLIR_GetTypeLLVMStructField(ctx, ty, i);
            unsigned fsz = llvm_type_size(ctx, ft, ptr_bytes);
            unsigned fal = llvm_type_align(ctx, ft, ptr_bytes);
            // A zero-length array (`int tail[0];`) is a valid size of 0.
            if ((fsz == 0 && !llvm_type_is_empty(ctx, ft, ptr_bytes)) || fal == 0)
                return 0;
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

// The indices of an `llvm.insertvalue` `position` attribute; the offset and
// type of an aggregate's element; the memory image of a global's constant
// initializer region. See mlir_llvm_global_init.c.
size_t llvm_position_indices(MLIR_Context *ctx, MLIR_AttributeHandle a,
                             int64_t *idx, size_t max);
bool llvm_aggregate_elem(MLIR_Context *ctx, MLIR_TypeHandle ty,
                         const int64_t *idx, size_t n, unsigned ptr_bytes,
                         unsigned *off, MLIR_TypeHandle *elem);
bool llvm_global_init_bytes(MLIR_Context *ctx, MLIR_OpHandle gop,
                            MLIR_TypeHandle gty, unsigned ptr_bytes,
                            uint8_t *buf, unsigned size);

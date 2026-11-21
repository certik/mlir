// Memory image of an `llvm.mlir.global`'s constant initializer region, for
// the backends that lay out data themselves (mlir_llvm_to_wasmssa.c,
// mlir_llvm_to_x64_emit.inc, mlir_llvm_to_aarch64.c) and the LLVM IR
// printer (mlir_translate_to_llvm_ir.c). Kept out of the header so tinyC,
// which emits every static inline body into each translation unit, compiles
// it once.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "mlir_api.h"
#include "mlir_llvm_type_layout.h"

// The indices of an `llvm.insertvalue` `position` attribute, printed as
// "array<i64: 1, 2>" (DenseI64ArrayAttr) or "[1, 2]". Returns how many there
// are, or 0 when there are none or more than `max`.
size_t llvm_position_indices(MLIR_Context *ctx, MLIR_AttributeHandle a,
                             int64_t *idx, size_t max) {
    string s = MLIR_GetAttributeAsString(ctx, a);
    const char *p = s.str;
    const char *end = s.str + s.size;
    // Skip "array<i64:" so the element width is not read as an index.
    for (const char *q = p; q < end; q++) {
        if (*q == ':') { p = q + 1; break; }
    }
    size_t n = 0;
    for (;;) {
        while (p < end && (*p < '0' || *p > '9')) p++;
        if (p >= end) return n;
        if (n == max) return 0;
        int64_t v = 0;
        while (p < end && *p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
        idx[n++] = v;
    }
}

// The byte offset and type of the element at position idx[0..n) inside the
// aggregate `ty`. False for a position that does not exist.
bool llvm_aggregate_elem(MLIR_Context *ctx, MLIR_TypeHandle ty,
                         const int64_t *idx, size_t n, unsigned ptr_bytes,
                         unsigned *off, MLIR_TypeHandle *elem) {
    unsigned o = 0;
    for (size_t k = 0; k < n; k++) {
        if (MLIR_IsTypeLLVMArray(ty)) {
            if ((uint64_t)idx[k] >= MLIR_GetTypeLLVMArrayNumElements(ty)) return false;
            ty = MLIR_GetTypeLLVMArrayElement(ty);
            o += (unsigned)idx[k] * llvm_type_size(ctx, ty, ptr_bytes);
        } else if (MLIR_IsTypeLLVMStruct(ty)) {
            if ((uint64_t)idx[k] >= MLIR_GetTypeLLVMStructNumFields(ty)) return false;
            o += llvm_struct_field_offset(ctx, ty, (size_t)idx[k], ptr_bytes);
            ty = MLIR_GetTypeLLVMStructField(ty, (size_t)idx[k]);
        } else {
            return false;
        }
    }
    *off = o;
    *elem = ty;
    return true;
}

// Fold the initializer region of the `llvm.mlir.global` `gop`, of type `gty`,
// into its little-endian memory image `buf` (`size` bytes, zero-filled by the
// caller). The region must start from an `llvm.mlir.zero` (or `undef` /
// `poison`) and insert `llvm.mlir.constant` scalars into it with a chain of
// `llvm.insertvalue`s that `llvm.return` returns: what tinyc emits for a
// struct initializer, and MLIR_CreateLLVMGlobalArrayInit's upstream
// implementation for an array. Returns false for any other region (e.g. one
// returning an address).
bool llvm_global_init_bytes(MLIR_Context *ctx, MLIR_OpHandle gop,
                            MLIR_TypeHandle gty, unsigned ptr_bytes,
                            uint8_t *buf, unsigned size) {
    if (MLIR_GetOpNumRegions(gop) == 0) return false;
    MLIR_RegionHandle rgn = MLIR_GetOpRegion(gop, 0);
    if (MLIR_GetRegionNumBlocks(rgn) == 0) return false;
    MLIR_BlockHandle blk = MLIR_GetRegionBlock(rgn, 0);
    size_t nb = MLIR_GetBlockNumOps(blk);
    MLIR_ValueHandle cur = MLIR_INVALID_HANDLE;  // the aggregate built so far
    for (size_t i = 0; i < nb; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(blk, i);
        MLIR_OpType t = MLIR_GetOpType(op);
        if (t == OP_TYPE_LLVM_MLIR_ZERO || t == OP_TYPE_LLVM_MLIR_UNDEF ||
            t == OP_TYPE_LLVM_MLIR_POISON) {
            if (cur != MLIR_INVALID_HANDLE || MLIR_GetOpNumResults(op) != 1) return false;
            cur = MLIR_GetOpResult(op, 0);
        } else if (t == OP_TYPE_LLVM_MLIR_CONSTANT) {
            // Read through the llvm.insertvalue that uses it.
        } else if (t == OP_TYPE_LLVM_INSERTVALUE) {
            if (cur == MLIR_INVALID_HANDLE || MLIR_GetOpNumOperands(op) != 2 ||
                MLIR_GetOpOperand(op, 0) != cur || MLIR_GetOpNumResults(op) != 1)
                return false;
            MLIR_AttributeHandle pa = MLIR_GetOpAttributeByName(op, "position");
            if (pa == MLIR_INVALID_HANDLE) return false;
            int64_t idx[16];
            size_t n = llvm_position_indices(ctx, pa, idx, 16);
            unsigned off = 0;
            MLIR_TypeHandle et = MLIR_INVALID_HANDLE;
            if (n == 0 || !llvm_aggregate_elem(ctx, gty, idx, n, ptr_bytes, &off, &et))
                return false;
            unsigned esz = llvm_type_size(ctx, et, ptr_bytes);
            if (esz == 0 || esz > 8 || off + esz > size) return false;
            MLIR_OpHandle c = MLIR_GetValueDefiningOp(MLIR_GetOpOperand(op, 1));
            if (c == MLIR_INVALID_HANDLE || MLIR_GetOpType(c) != OP_TYPE_LLVM_MLIR_CONSTANT)
                return false;
            MLIR_AttributeHandle va = MLIR_GetOpAttributeByName(c, "value");
            if (va == MLIR_INVALID_HANDLE) return false;
            MLIR_AttrKind k = MLIR_GetAttributeKind(va);
            uint64_t bits = 0;
            if (k == MLIR_ATTR_KIND_INTEGER || k == MLIR_ATTR_KIND_BOOL) {
                bits = (uint64_t)MLIR_GetAttributeInteger(va);
            } else if (k == MLIR_ATTR_KIND_FLOAT && esz == 4) {
                float f = (float)MLIR_GetAttributeFloat(va);
                uint32_t b32;
                memcpy(&b32, &f, 4);
                bits = b32;
            } else if (k == MLIR_ATTR_KIND_FLOAT && esz == 8) {
                double d = MLIR_GetAttributeFloat(va);
                memcpy(&bits, &d, 8);
            } else {
                return false;
            }
            for (unsigned b = 0; b < esz; b++) buf[off + b] = (uint8_t)(bits >> (8 * b));
            cur = MLIR_GetOpResult(op, 0);
        } else if (t == OP_TYPE_LLVM_RETURN) {
            return cur != MLIR_INVALID_HANDLE && MLIR_GetOpNumOperands(op) == 1 &&
                   MLIR_GetOpOperand(op, 0) == cur;
        } else {
            return false;
        }
    }
    return false;
}

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

// The indices of an `llvm.insertvalue` `position` attribute (a dense i64
// array, or an array of integers). Returns how many there are, or 0 when
// there are none or more than `max`.
size_t llvm_position_indices(MLIR_Context *ctx, MLIR_AttributeHandle a,
                             int64_t *idx, size_t max) {
    if (MLIR_GetAttributeKind(ctx, a) == MLIR_ATTR_KIND_ARRAY) {
        size_t n = MLIR_GetAttributeArraySize(ctx, a);
        if (n > max) return 0;
        for (size_t i = 0; i < n; i++)
            idx[i] = MLIR_GetAttributeInteger(ctx, MLIR_GetAttributeArrayElement(ctx, a, i));
        return n;
    }
    size_t n = MLIR_GetAttributeDenseSize(ctx, a);
    if (n > max) return 0;
    for (size_t i = 0; i < n; i++) idx[i] = MLIR_GetAttributeDenseElement(ctx, a, i);
    return n;
}

// The byte offset and type of the element at position idx[0..n) inside the
// aggregate `ty`. False for a position that does not exist.
bool llvm_aggregate_elem(MLIR_Context *ctx, MLIR_TypeHandle ty,
                         const int64_t *idx, size_t n, unsigned ptr_bytes,
                         unsigned *off, MLIR_TypeHandle *elem) {
    unsigned o = 0;
    for (size_t k = 0; k < n; k++) {
        if (MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_LLVM_ARRAY) {
            if ((uint64_t)idx[k] >= MLIR_GetTypeNumElements(ctx, ty)) return false;
            ty = MLIR_GetTypeElement(ctx, ty);
            o += (unsigned)idx[k] * llvm_type_size(ctx, ty, ptr_bytes);
        } else if (MLIR_GetTypeKind(ctx, ty) == MLIR_TYPE_KIND_LLVM_STRUCT) {
            if ((uint64_t)idx[k] >= MLIR_GetTypeLLVMStructNumFields(ctx, ty)) return false;
            o += llvm_struct_field_offset(ctx, ty, (size_t)idx[k], ptr_bytes);
            ty = MLIR_GetTypeLLVMStructField(ctx, ty, (size_t)idx[k]);
        } else {
            return false;
        }
    }
    *off = o;
    *elem = ty;
    return true;
}

// The last struct field located by llvm_global_init_bytes: a chain of
// insertvalues into consecutive fields (as tinyc emits them, sorted by
// position) then costs O(1) per field instead of O(field index).
typedef struct {
    bool valid;
    int64_t prefix[16];         // position of the struct
    size_t n;                   // its length
    MLIR_TypeHandle sty;        // the struct type
    unsigned soff;              // its offset
    size_t fld;                 // the field
    unsigned foff;              // the field's offset in the struct
} FieldCursor;

// llvm_aggregate_elem, continuing from `fc` when idx is a later field of
// the same struct.
static bool elem_from_cursor(MLIR_Context *ctx, MLIR_TypeHandle gty, const int64_t *idx,
                             size_t n, unsigned ptr_bytes, FieldCursor *fc,
                             unsigned *off, MLIR_TypeHandle *elem) {
    bool same = fc->valid && n == fc->n + 1 && (uint64_t)idx[n - 1] >= fc->fld;
    for (size_t k = 0; same && k + 1 < n; k++) {
        if (idx[k] != fc->prefix[k]) same = false;
    }
    if (!same) {
        // Locate the parent; continue from its first field if a struct.
        MLIR_TypeHandle pty;
        unsigned poff;
        if (!llvm_aggregate_elem(ctx, gty, idx, n - 1, ptr_bytes, &poff, &pty)) return false;
        if (MLIR_GetTypeKind(ctx, pty) != MLIR_TYPE_KIND_LLVM_STRUCT)
            return llvm_aggregate_elem(ctx, gty, idx, n, ptr_bytes, off, elem);
        fc->valid = true;
        for (size_t k = 0; k + 1 < n; k++) fc->prefix[k] = idx[k];
        fc->n = n - 1;
        fc->sty = pty;
        fc->soff = poff;
        fc->fld = 0;
        fc->foff = 0;
    }
    size_t nf = MLIR_GetTypeLLVMStructNumFields(ctx, fc->sty);
    if ((uint64_t)idx[n - 1] >= nf) return false;
    while (fc->fld < (size_t)idx[n - 1]) {
        MLIR_TypeHandle ft = MLIR_GetTypeLLVMStructField(ctx, fc->sty, fc->fld);
        MLIR_TypeHandle nt = MLIR_GetTypeLLVMStructField(ctx, fc->sty, fc->fld + 1);
        fc->foff = llvm_align_up(fc->foff + llvm_type_size(ctx, ft, ptr_bytes),
                                 llvm_type_align(ctx, nt, ptr_bytes));
        fc->fld++;
    }
    *off = fc->soff + fc->foff;
    *elem = MLIR_GetTypeLLVMStructField(ctx, fc->sty, fc->fld);
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
bool llvm_global_init_bytes_relocs(MLIR_Context *ctx, MLIR_OpHandle gop,
                                   MLIR_TypeHandle gty, unsigned ptr_bytes,
                                   uint8_t *buf, unsigned size,
                                   LlvmGlobalRelocFn on_reloc, void *ud) {
    if (MLIR_GetOpNumRegions(ctx, gop) == 0) return false;
    MLIR_RegionHandle rgn = MLIR_GetOpRegion(ctx, gop, 0);
    if (MLIR_GetRegionNumBlocks(ctx, rgn) == 0) return false;
    MLIR_BlockHandle blk = MLIR_GetRegionBlock(ctx, rgn, 0);
    size_t nb = MLIR_GetBlockNumOps(ctx, blk);
    MLIR_ValueHandle cur = MLIR_INVALID_HANDLE;  // the aggregate built so far
    FieldCursor fc;
    fc.valid = false;
    for (size_t i = 0; i < nb; i++) {
        MLIR_OpHandle op = MLIR_GetBlockOp(ctx, blk, i);
        MLIR_OpType t = MLIR_GetOpType(ctx, op);
        if (t == OP_TYPE_LLVM_MLIR_ZERO || t == OP_TYPE_LLVM_MLIR_UNDEF ||
            t == OP_TYPE_LLVM_MLIR_POISON) {
            if (cur != MLIR_INVALID_HANDLE || MLIR_GetOpNumResults(ctx, op) != 1) return false;
            cur = MLIR_GetOpResult(ctx, op, 0);
        } else if (t == OP_TYPE_LLVM_MLIR_CONSTANT ||
                   t == OP_TYPE_LLVM_MLIR_ADDRESSOF) {
            // Read through the llvm.insertvalue that uses it.
        } else if (t == OP_TYPE_LLVM_INSERTVALUE) {
            if (cur == MLIR_INVALID_HANDLE || MLIR_GetOpNumOperands(ctx, op) != 2 ||
                MLIR_GetOpOperand(ctx, op, 0) != cur || MLIR_GetOpNumResults(ctx, op) != 1)
                return false;
            MLIR_AttributeHandle pa = MLIR_GetOpAttr(ctx, op, MLIR_KEY_POSITION);
            if (pa == MLIR_INVALID_HANDLE) return false;
            int64_t idx[16];
            size_t n = llvm_position_indices(ctx, pa, idx, 16);
            unsigned off = 0;
            MLIR_TypeHandle et = MLIR_INVALID_HANDLE;
            if (n == 0 || !elem_from_cursor(ctx, gty, idx, n, ptr_bytes, &fc, &off, &et))
                return false;
            unsigned esz = llvm_type_size(ctx, et, ptr_bytes);
            if (esz == 0 || esz > 8 || off + esz > size) return false;
            MLIR_OpHandle c = MLIR_GetValueDefiningOp(ctx, MLIR_GetOpOperand(ctx, op, 1));
            if (c == MLIR_INVALID_HANDLE) return false;
            MLIR_OpType ct = MLIR_GetOpType(ctx, c);
            if (ct == OP_TYPE_LLVM_MLIR_ADDRESSOF) {
                if (!on_reloc || esz != ptr_bytes) return false;
                MLIR_AttributeHandle ga = MLIR_GetOpAttr(ctx, c, MLIR_KEY_GLOBAL_NAME);
                if (ga == MLIR_INVALID_HANDLE)
                    ga = MLIR_GetOpAttr(ctx, c, MLIR_KEY_VALUE);
                if (ga == MLIR_INVALID_HANDLE) return false;
                string ts = MLIR_GetAttributeAsString(ctx, ga);
                if (ts.size > 0 && ts.str[0] == '@') { ts.str++; ts.size--; }
                if (ts.size == 0) return false;
                on_reloc(ud, off, ts);
                cur = MLIR_GetOpResult(ctx, op, 0);
                continue;
            }
            if (ct != OP_TYPE_LLVM_MLIR_CONSTANT) return false;
            MLIR_AttributeHandle va = MLIR_GetOpAttr(ctx, c, MLIR_KEY_VALUE);
            if (va == MLIR_INVALID_HANDLE) return false;
            MLIR_AttrKind k = MLIR_GetAttributeKind(ctx, va);
            uint64_t bits = 0;
            if (k == MLIR_ATTR_KIND_INTEGER || k == MLIR_ATTR_KIND_BOOL) {
                bits = (uint64_t)MLIR_GetAttributeInteger(ctx, va);
            } else if (k == MLIR_ATTR_KIND_FLOAT && esz == 4) {
                float f = (float)MLIR_GetAttributeFloat(ctx, va);
                uint32_t b32;
                memcpy(&b32, &f, 4);
                bits = b32;
            } else if (k == MLIR_ATTR_KIND_FLOAT && esz == 8) {
                double d = MLIR_GetAttributeFloat(ctx, va);
                memcpy(&bits, &d, 8);
            } else {
                return false;
            }
            for (unsigned b = 0; b < esz; b++) buf[off + b] = (uint8_t)(bits >> (8 * b));
            cur = MLIR_GetOpResult(ctx, op, 0);
        } else if (t == OP_TYPE_LLVM_RETURN) {
            return cur != MLIR_INVALID_HANDLE && MLIR_GetOpNumOperands(ctx, op) == 1 &&
                   MLIR_GetOpOperand(ctx, op, 0) == cur;
        } else {
            return false;
        }
    }
    return false;
}

bool llvm_global_init_bytes(MLIR_Context *ctx, MLIR_OpHandle gop,
                            MLIR_TypeHandle gty, unsigned ptr_bytes,
                            uint8_t *buf, unsigned size) {
    return llvm_global_init_bytes_relocs(ctx, gop, gty, ptr_bytes, buf, size, NULL, NULL);
}

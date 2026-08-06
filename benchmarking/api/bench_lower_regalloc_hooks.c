// Minimal regalloc hooks for the x64 lowering benchmark link.

#include <string.h>

#include "mlir_api.h"
#include "mlir_regalloc.h"

static bool name_eq(string s, const char *cstr) {
    size_t n = strlen(cstr);
    return s.size == n && !memcmp(s.str, cstr, n);
}

bool const_int_val(MLIR_Context *ctx, MLIR_OpHandle op, int64_t *val,
                   uint8_t *is64) {
    if (!name_eq(MLIR_GetOpName(op), "llvm.mlir.constant")) return false;
    MLIR_ValueHandle res = MLIR_GetOpResult(op, 0);
    string ty = MLIR_GetTypeString(ctx, MLIR_GetValueType(res));
    if (ty.size == 3 && !memcmp(ty.str, "f32", 3)) return false;
    if (ty.size == 3 && !memcmp(ty.str, "f64", 3)) return false;
    MLIR_AttributeHandle va = MLIR_GetOpAttributeByName(op, "value");
    if (va == MLIR_INVALID_HANDLE) return false;
    *val = MLIR_GetAttributeInteger(va);
    *is64 = (ty.size == 3 && !memcmp(ty.str, "i64", 3)) ||
            (ty.size >= 5 && !memcmp(ty.str, "!llvm", 5)) ||
            (ty.size == 3 && !memcmp(ty.str, "ptr", 3));
    return true;
}

bool cast_src(MLIR_OpHandle op, MLIR_ValueHandle *src) {
    string nm = MLIR_GetOpName(op);
    if (!name_eq(nm, "llvm.inttoptr") && !name_eq(nm, "llvm.ptrtoint") &&
        !name_eq(nm, "arith.index_cast") && !name_eq(nm, "arith.index_castui") &&
        !name_eq(nm, "llvm.trunc") && !name_eq(nm, "llvm.zext") &&
        !name_eq(nm, "llvm.sext") && !name_eq(nm, "llvm.bitcast"))
        return false;
    *src = MLIR_GetOpOperand(op, 0);
    return true;
}

bool a64_is_spine(MLIR_OpHandle op, SlotMap *memspine) {
    (void)op; (void)memspine; return false;
}
bool a64_memfuse_uses(MLIR_Context *ctx, MLIR_OpHandle op, SlotMap *memfuse,
                      MLIR_ValueHandle *base, MLIR_ValueHandle *idx) {
    (void)ctx; (void)op; (void)memfuse; (void)base; (void)idx; return false;
}
bool a64_shiftfuse_uses(MLIR_Context *ctx, MLIR_OpHandle op,
                        SlotMap *shiftfuse, MLIR_ValueHandle *xv) {
    (void)ctx; (void)op; (void)shiftfuse; (void)xv; return false;
}

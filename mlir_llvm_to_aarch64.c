// llvm dialect -> aarch64 lowering. See mlir_llvm_to_aarch64.h for the
// public API, rationale, and the staged build-up plan.
//
// Status: Step 0 scaffold. The entry point is wired into the driver behind
// the opt-in `--macho-backend=llvm` flag but does not yet lower anything;
// it returns a clear "not implemented" diagnostic. Subsequent steps add:
//   Step 1  walking skeleton: int main(){return 42;}
//   Step 2  grow instruction selection to pass the native suite
//   Step 3  post-isel register allocation with operand constraints
// Nothing here is on the path of the existing wasm/wmir backends.

#include <stdio.h>

#include "mlir_llvm_to_aarch64.h"

MLIR_OpHandle mlir_llvm_to_aarch64(MLIR_Context *ctx,
                                   MLIR_OpHandle llvm_module) {
    (void)ctx;
    (void)llvm_module;
    fprintf(stderr,
        "llvm->aarch64: backend not implemented yet "
        "(--macho-backend=llvm is a work-in-progress scaffold; "
        "use --macho-backend=wmir or the default wasm backend)\n");
    return MLIR_INVALID_HANDLE;
}

#pragma once

#include "mlir_api.h"
#include "mlir_passes.h"

#ifdef __cplusplus
extern "C" {
#endif

// C port of LFortran's `recombine_horizontal_reductions` LLVM optimization pass
// built purely on the public MLIR C API (mlir_api.h + mlir_passes.h).
MLIR_Pass MLIR_PassLLVMReductions(void);

#ifdef __cplusplus
}
#endif

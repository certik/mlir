// Text printers for IR built with the core API (mlir_api.h).
//
// MLIR_PrintOperationClassic (mlir_classic_printer.c) and
// MLIR_PrintOperationGeneric (mlir_generic_printer.c) each live in their
// own translation unit, so the linker pulls in only what a binary calls.
// The upstream printer, MLIR_PrintOperationUpstream, is in
// mlir_api_upstream.h.

#pragma once

#include "mlir_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Print an operation in the classic (custom assembly) form or in the
// generic form.
string MLIR_PrintOperationClassic(MLIR_Context *ctx, MLIR_OpHandle op);
string MLIR_PrintOperationGeneric(MLIR_Context *ctx, MLIR_OpHandle op);

// -----------------------------------------------------------------------------
// Printer support
// -----------------------------------------------------------------------------
//
// Functions that only the printers (and the parser, mlir_parser.h) use.
// Each backend implements them; they are not part of the core API.

// Named location references (`#locN = loc(...)`) collected by the parser
// (mlir_parser.c implements these two functions) and printed back by the
// classic printer. Declared here rather than in mlir_parser.h so that the
// printers do not include the parser's header.
typedef struct MLIR_LocationMap {
    void *impl;
} MLIR_LocationMap;

size_t MLIR_GetLocationMapSize(const MLIR_LocationMap *location_map);
size_t MLIR_CollectLocationMap(const MLIR_LocationMap *location_map, string *out_keys, MLIR_LocationHandle *out_locs, size_t max);

// SSA name the value was created with (e.g. "%0"), or an empty string.
string MLIR_GetValueRegisterName(MLIR_ValueHandle value);
// Source text a location was parsed from, or an empty string (always
// empty on the upstream backend).
string MLIR_GetLocationOriginalText(MLIR_LocationHandle loc);
// Returns the 0-indexed position of `block` within its parent region.
// Returns SIZE_MAX if the block has no parent or the backend doesn't track it.
size_t MLIR_GetBlockIndex(MLIR_BlockHandle block);

#ifdef __cplusplus
}
#endif

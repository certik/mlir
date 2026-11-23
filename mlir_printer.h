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
string MLIR_GetValueRegisterName(MLIR_Context *ctx, MLIR_ValueHandle value);
// Source text a location was parsed from, or an empty string (always
// empty on the upstream backend).
string MLIR_GetLocationOriginalText(MLIR_LocationHandle loc);
// Returns the 0-indexed position of `block` within its parent region.
// Returns SIZE_MAX if the block has no parent or the backend doesn't track it.
size_t MLIR_GetBlockIndex(MLIR_Context *ctx, MLIR_BlockHandle block);

// -----------------------------------------------------------------------------
// Parser-only ops
// -----------------------------------------------------------------------------
//
// The ops of the dialects that only the text parser and the classic printer
// know (tt, gpu, affine, vector, std, tensor, linalg, index, math, memref:
// MLIR_PARSER_OP and MLIR_PRINTER_OP in mlir_ops.def) have no static MLIR_OpType
// (doc/mlir_api_design.md, decision D9). The parser and the printers
// register their names as dynamic op types of the context
// (parser_op_types_init) and dispatch on an op kind (parser_op_kind): the
// MLIR_OpType for a static type, a PARSER_OP_* value for a parser-only op,
// and OP_TYPE_UNREGISTERED for any other dynamic type. The PARSER_OP_*
// values start at OP_TYPE_COUNT, so one switch can mix them with OP_TYPE_*
// cases.

typedef enum {
    PARSER_OP_BEGIN_ = OP_TYPE_COUNT - 1,
#define MLIR_PARSER_OP(kind, name) kind,
#include "mlir_ops.def"
    // From here on, the ops that only the classic printer recognizes.
    PARSER_OP_PRINTER_ONLY_,
    PARSER_OP_PRINTER_ONLY_BEGIN_ = PARSER_OP_PRINTER_ONLY_ - 1,
#define MLIR_PRINTER_OP(kind, name) kind,
#include "mlir_ops.def"
    PARSER_OP_END_
} ParserOpKind;

// The capacity of ParserOpTypes.type: a literal because tinyC accepts no
// enumerator as an array length.
#define PARSER_OP_CAPACITY 48
#ifndef __TINYC__
typedef char parser_op_capacity_ok[
    ((int)PARSER_OP_END_ - (int)OP_TYPE_COUNT <= PARSER_OP_CAPACITY) ? 1 : -1];
#endif

typedef struct ParserOpTypes {
    // type[k - OP_TYPE_COUNT] is the dynamic type of kind k.
    MLIR_OpType type[PARSER_OP_CAPACITY];
    // True if type[i] == type[0] + i for every kind, which is the case
    // unless one of the names was registered on its own before; then
    // parser_op_kind is a subtraction rather than a search.
    bool consecutive;
} ParserOpTypes;

// Registers the parser-only op names in `ctx` and records their types
// (mlir_op_names.c, which every build that links a printer links too).
void parser_op_types_init(MLIR_Context *ctx, ParserOpTypes *p);
// The op kind of type `t` (see above).
int parser_op_kind(const ParserOpTypes *p, MLIR_OpType t);

#ifdef __cplusplus
}
#endif

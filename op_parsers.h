/*
This file contains specific operation parsers.

Each parser is using Parser to parse a specific operation and return it in `op`.
*/
#pragma once

#include <base/arena.h>
#include <base/string.h>
#include "mlir_parser.h"

#ifdef __cplusplus
extern "C" {
#endif

// Parsing functions for specific operation types
void parse_gpu_launch(Parser *parser, Operation *op);
void parse_tt_func(Parser *parser, Operation *op);
void parse_scf_if(Parser *parser, Operation *op);
void parse_scf_for(Parser *parser, Operation *op);
void parse_scf_while(Parser *parser, Operation *op);
void parse_scf_yield(Parser *parser, Operation *op);
void parse_return_operation(Parser *parser, Operation *op);

// Static parsing functions (exposed for extraction)
void parse_arith_constant(Parser *parser, Operation *op);
void parse_arith_cmpi(Parser *parser, Operation *op);
void parse_arith_binary(Parser *parser, Operation *op);
void parse_arith_select(Parser *parser, Operation *op);
void parse_tt_get_program_id(Parser *parser, Operation *op);
void parse_tt_splat(Parser *parser, Operation *op);
void parse_tt_make_range(Parser *parser, Operation *op);
void parse_tt_addptr_load_store(Parser *parser, Operation *op);
void parse_tt_store(Parser *parser, Operation *op);
void parse_tt_call(Parser *parser, Operation *op);
void parse_func_func(Parser *parser, Operation *op);
void parse_func_call(Parser *parser, Operation *op);
void parse_affine_for(Parser *parser, Operation *op);
void parse_memref_load_or_store(Parser *parser, Operation *op);
void parse_memref_store(Parser *parser, Operation *op);
void parse_vector_print(Parser *parser, Operation *op);
void parse_std_constant(Parser *parser, Operation *op);
void parse_tt_reduce(Parser *parser, Operation *op);
void parse_tensor_extract(Parser *parser, Operation *op);
void parse_cf_br(Parser *parser, Operation *op);
void parse_linalg_fill(Parser *parser, Operation *op);
void parse_affine_load(Parser *parser, Operation *op);
void parse_index_constant(Parser *parser, Operation *op);
void parse_tensor_splat(Parser *parser, Operation *op);
void parse_tensor_collapse_shape(Parser *parser, Operation *op);
void parse_generic_operation(Parser *parser, Operation *op);
void parse_generic_attrs_and_result_type(Parser *parser, Operation *op);

// Helper function
void consume_optional_hash_selector(Parser *parser);

#ifdef __cplusplus
}
#endif

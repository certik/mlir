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

// Specific operation parsers

void parse_affine_load(Parser *parser, MlirOperation *op);
void parse_affine_for(Parser *parser, MlirOperation *op);

void parse_arith_binary(Parser *parser, MlirOperation *op);
void parse_arith_cmpi(Parser *parser, MlirOperation *op);
OperationParserResult parse_arith_constant_op(const OperationParserParams *params);
void parse_arith_select(Parser *parser, MlirOperation *op);

void parse_cf_br(Parser *parser, MlirOperation *op);
void parse_cf_cond_br(Parser *parser, MlirOperation *op);

void parse_func_call(Parser *parser, MlirOperation *op);
void parse_func_func(Parser *parser, MlirOperation *op);

void parse_gpu_launch(Parser *parser, MlirOperation *op);

void parse_index_constant(Parser *parser, MlirOperation *op);

void parse_linalg_fill(Parser *parser, MlirOperation *op);

OperationParserResult parse_memref_load_op(const OperationParserParams *params);
OperationParserResult parse_memref_store_op(const OperationParserParams *params);
void parse_memref_load_or_store(Parser *parser, MlirOperation *op);
void parse_memref_store(Parser *parser, MlirOperation *op);

void parse_return_operation(Parser *parser, MlirOperation *op);

void parse_scf_if(Parser *parser, MlirOperation *op);
void parse_scf_for(Parser *parser, MlirOperation *op);
void parse_scf_while(Parser *parser, MlirOperation *op);
void parse_scf_yield(Parser *parser, MlirOperation *op);

void parse_std_constant(Parser *parser, MlirOperation *op);

void parse_tensor_collapse_shape(Parser *parser, MlirOperation *op);
void parse_tensor_extract(Parser *parser, MlirOperation *op);
void parse_tensor_splat(Parser *parser, MlirOperation *op);

void parse_tt_addptr_load_store(Parser *parser, MlirOperation *op);
void parse_tt_call(Parser *parser, MlirOperation *op);
void parse_tt_func(Parser *parser, MlirOperation *op);
void parse_tt_get_program_id(Parser *parser, MlirOperation *op);
void parse_tt_make_range(Parser *parser, MlirOperation *op);
void parse_tt_reduce(Parser *parser, MlirOperation *op);
void parse_tt_splat(Parser *parser, MlirOperation *op);
void parse_tt_store(Parser *parser, MlirOperation *op);

void parse_vector_print(Parser *parser, MlirOperation *op);


// Helper functions
void consume_optional_hash_selector(Parser *parser);
void parse_generic_attrs_and_result_type(Parser *parser, MlirOperation *op);
void parse_generic_operation(Parser *parser, MlirOperation *op);

#ifdef __cplusplus
}
#endif

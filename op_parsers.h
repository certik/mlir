/*
This file contains specific operation parsers.

Each parser is using Parser to parse a specific operation and return it in `op`.
*/
#pragma once

#include <base/arena.h>
#include <base/string.h>
#include "mlir_api.h"
#include "mlir_parser.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OperationParserParams {
    Arena  *arena;
    OpType  op_type;
    string  opname; /* only non-empty for unregistered ops */

    MlirValue **lhs_results;
    size_t     n_lhs_results;

    MlirLocation *unnumbered_loc_def;
    int64_t       source_line_start;
} OperationParserParams;

typedef struct OperationParserResult {
    MlirOperation *operation;
    MlirValue    **results;
    size_t         n_results;
    MlirLocation  *location; /* may be NULL */
} OperationParserResult;

// Specific operation parsers

OperationParserResult parse_affine_load_op(Parser *parser, const OperationParserParams *params);
void parse_affine_for(Parser *parser, MlirOperation *op);

OperationParserResult parse_arith_binary_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_arith_cmpi_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_arith_constant_op(Parser *parser, OperationParserParams *params);
OperationParserResult parse_arith_select_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_cf_br_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_cf_cond_br_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_func_call_op(Parser *parser, const OperationParserParams *params);
void parse_func_func(Parser *parser, MlirOperation *op);

OperationParserResult parse_gpu_launch_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_index_constant_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_linalg_fill_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_memref_load_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_memref_store_op(Parser *parser, const OperationParserParams *params);

void parse_return_operation(Parser *parser, MlirOperation *op);

void parse_scf_if(Parser *parser, MlirOperation *op);
void parse_scf_for(Parser *parser, MlirOperation *op);
void parse_scf_while(Parser *parser, MlirOperation *op);
void parse_scf_yield(Parser *parser, MlirOperation *op);

OperationParserResult parse_std_constant_op(Parser *parser, const OperationParserParams *params);

OperationParserResult parse_tensor_collapse_shape_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tensor_extract_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tensor_splat_op(Parser *parser, const OperationParserParams *params);

void parse_tt_addptr_load_store(Parser *parser, MlirOperation *op);
OperationParserResult parse_tt_call_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tt_func_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tt_get_program_id_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tt_make_range_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tt_reduce_op(Parser *parser, const OperationParserParams *params);
OperationParserResult parse_tt_splat_op(Parser *parser, const OperationParserParams *params);
void parse_tt_store(Parser *parser, MlirOperation *op);

OperationParserResult parse_vector_print_op(Parser *parser, const OperationParserParams *params);


// Helper functions
void consume_optional_hash_selector(Parser *parser);
void parse_generic_attrs_and_result_type(Parser *parser, MlirOperation *op);
void parse_generic_operation(Parser *parser, MlirOperation *op);

#ifdef __cplusplus
}
#endif

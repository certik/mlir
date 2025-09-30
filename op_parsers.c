#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <base/string.h>
#include <base/io.h>
#include <base/vector.h>

#include "mlir_parser.h"
#include "tokenizer.h"
#include "op_parsers.h"


OperationParserResult parse_arith_constant_op(Parser *parser, OperationParserParams *params) {
    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    MlirType **result_types = NULL;
    size_t n_result_types = 0;

    MlirValue **results = params->lhs_results;
    size_t n_results = params->n_lhs_results;
    MlirValue *result_value = (n_results > 0) ? results[0] : NULL;

    MlirLocation *op_location = NULL;
    bool has_explicit_type = false;

    if (parser_peek(parser, TK_INTEGER)) {
        string value_str = parser_token_str(parser);
        parser_expect(parser, TK_INTEGER);

        int64_t parsed_value = 0;
        for (size_t i = 0; i < value_str.size; i++) {
            if (value_str.str[i] >= '0' && value_str.str[i] <= '9') {
                parsed_value = parsed_value * 10 + (value_str.str[i] - '0');
            }
        }
        append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                    create_integer_attr(parser, str_lit("value"), parsed_value));
    } else if (parser_peek(parser, TK_REAL)) {
        string value_str = parser_token_str(parser);
        parser_expect(parser, TK_REAL);

        char *buffer = arena_alloc_array(parser->arena, char, value_str.size + 1);
        memcpy(buffer, value_str.str, value_str.size);
        buffer[value_str.size] = '\0';
        double parsed_value = strtod(buffer, NULL);
        append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                    create_float_attr(parser, str_lit("value"), parsed_value));
    } else if (parser_peek(parser, TK_NAME)) {
        string name_str = parser_token_str(parser);
        if (str_eq(name_str, str_lit("true")) || str_eq(name_str, str_lit("false"))) {
            parser_expect(parser, TK_NAME);
            append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                        create_integer_attr(parser, str_lit("value"), str_eq(name_str, str_lit("true")) ? 1 : 0));

            if (!result_value) {
                result_value = mlir_value_create(parser->arena, OP_RESULT);
            }
            MlirType *bool_type = mlir_type_create_from_string(parser->arena, str_lit("i1"));
            result_types = arena_alloc_array(parser->arena, MlirType*, 1);
            result_types[0] = bool_type;
            n_result_types = 1;
            mlir_value_set_type(result_value, bool_type);
        } else {
            string payload = str_lit("");
            int angle = 0;
            do {
                string tok = parser_token_str(parser);
                payload = payload.size ? str_concat(parser->arena, payload, tok) : tok;
                parser_next_token(parser);
                if (parser_peek(parser, TK_LANGLE)) angle++;
                else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
                if (angle == 0 && parser_peek(parser, TK_COLON)) break;
            } while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE));
            append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                        create_string_attr(parser, str_lit("value_text"), payload));
        }
    } else {
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_COLON)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            parser_next_token(parser);
        }
    }

    if (!has_explicit_type && parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        string type_str = str_lit("");
        if (parse_type_string(parser, &type_str)) {
            MlirType *type = mlir_type_create_from_string(parser->arena, type_str);
            if (!result_types) {
                result_types = arena_alloc_array(parser->arena, MlirType*, 1);
            }
            result_types[0] = type;
            n_result_types = 1;
            if (!result_value) {
                result_value = mlir_value_create(parser->arena, OP_RESULT);
            }
            mlir_value_set_type(result_value, type);
            has_explicit_type = true;
        }
    }

    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op_location = parse_loc(parser);
    }

    if (!result_value) {
        result_value = mlir_value_create(parser->arena, OP_RESULT);
    }

    if (!results) {
        results = arena_alloc_array(parser->arena, MlirValue*, 1);
        results[0] = result_value;
        n_results = 1;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        params->opname,
        attributes, n_attributes,
        result_types, n_result_types,
        results, n_results,
        NULL, 0,
        NULL, 0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    for (size_t i = 0; i < n_results; i++) {
        if (results[i]) {
            mlir_value_set_def(results[i], op);
        }
    }

    parse_generic_attrs_and_result_type(parser, op);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = mlir_operation_get_location(op)
    };
    return out;
}

OperationParserResult parse_arith_binary_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);

    if (parser_peek(parser, TK_REGISTER)) {
        if (!parse_register_operand(parser, &operands, false)) {
            OperationParserResult empty = {0};
            return empty;
        }

        if (parser_peek(parser, TK_COMMA)) {
            parser_expect(parser, TK_COMMA);
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str2 = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *operand2 = symbol_table_lookup(&parser->symbol_table, reg_str2);
                if (!operand2) {
                    parser_warning(parser, format(parser->arena, str_lit("Undefined value: {}"), reg_str2), parser->first, parser->last);
                    operand2 = mlir_value_create(parser->arena, BLOCK_ARG);
                    mlir_value_set_register_name(operand2, reg_str2.str, reg_str2.size);
                    MlirType *unknown_type = mlir_type_create_from_string(parser->arena, str_lit("unknown"));
                    mlir_value_set_type(operand2, unknown_type);
                }
                VecValue_push_back(parser->arena, &operands, operand2);
            }
        }
    }

    // Parse any attributes and result types using helper functions
    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0; // tracks attribute list growth for arith binary ops

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirType **result_types = NULL;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, NULL);

    // Optional default result type for specific ops when not provided
    if (n_result_types == 0 && params->op_type == OP_TYPE_ARITH_ADDF) {
        result_types = arena_alloc_array(params->arena, MlirType*, 1);
        result_types[0] = mlir_type_create_from_string(params->arena, str_lit("tensor<16xf32>"));
        n_result_types = 1;
    }

    // Parse optional location
    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // Create the operation at the end
    MlirOperation *op = mlir_op_create(params->arena, params->op_type, str_lit(""),
                                      attributes, n_attributes,
                                      result_types, n_result_types,
                                      params->lhs_results, params->n_lhs_results,
                                      operands.data, operands.size,
                                      NULL, 0,
                                      op_location, params->unnumbered_loc_def,
                                      str_lit(""), params->source_line_start);

    // Create results from the operation's result types
    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_func_call_op(Parser *parser, const OperationParserParams *params) {
    // Parse call @function(%args) : (arg_types) -> result_type
    // Function name (@name is tokenized as TK_FUNCTION_NAME)
    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    if (parser_peek(parser, TK_FUNCTION_NAME)) {
        // Capture callee
        string fname = parser_token_str(parser);
        parser_expect(parser, TK_FUNCTION_NAME);
        // Store as attribute 'callee'
        append_attr(parser, &attributes, &n_attributes, &attributes_capacity, create_string_attr(parser, str_lit("callee"), fname));
    }

    // Parse operands in parentheses
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);

        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!operand) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    OperationParserResult empty = {0};
                    return empty;
                }
                VecValue_push_back(parser->arena, &operands, operand);
            } else {
                parser_next_token(parser); // skip unknown tokens
            }

            if (parser_peek(parser, TK_COMMA)) {
                parser_expect(parser, TK_COMMA);
            } else if (!parser_peek(parser, TK_RPAREN)) {
                break;
            }
        }

        if (parser_peek(parser, TK_RPAREN)) {
            parser_expect(parser, TK_RPAREN);
        }
    }

    // Parse any additional attributes and result types
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirType **result_types = NULL;
    size_t n_result_types = 0;

    // Parse function signature: : (arg_types) -> result_type
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        // Skip arg types in parentheses
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            int paren_depth = 1;
            while (paren_depth > 0 && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LPAREN)) paren_depth++;
                else if (parser_peek(parser, TK_RPAREN)) paren_depth--;
                parser_next_token(parser);
            }
        }

        // Parse arrow and result type
        if (parser_peek(parser, TK_ARROW)) {
            parser_expect(parser, TK_ARROW);
            string type_str = str_lit("");
            if (parse_type_string(parser, &type_str)) {
                result_types = arena_alloc_array(params->arena, MlirType*, 1);
                result_types[0] = mlir_type_create_from_string(params->arena, type_str);
                n_result_types = 1;
            }
        }
    }

    // Parse any remaining result types if not already parsed
    if (n_result_types == 0) {
        parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, NULL);
    }

    // Parse optional location
    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // Create the operation at the end
    MlirOperation *op = mlir_op_create(params->arena, params->op_type, str_lit(""),
                                      attributes, n_attributes,
                                      result_types, n_result_types,
                                      params->lhs_results, params->n_lhs_results,
                                      operands.data, operands.size,
                                      NULL, 0,
                                      op_location, params->unnumbered_loc_def,
                                      str_lit(""), params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_tt_get_program_id_op(Parser *parser, const OperationParserParams *params) {
    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    if (parser_peek(parser, TK_NAME)) {
        string axis = parser_token_str(parser);
        parser_expect(parser, TK_NAME);
        append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                    create_string_attr(parser, str_lit("axis"), axis));
    }

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirType **result_types = NULL;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity,
                      params->op_type, NULL);

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        NULL,
        0,
        NULL,
        0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_tt_splat_op(Parser *parser, const OperationParserParams *params) {
    MlirValue **operands = NULL;
    size_t n_operands = 0;
    MlirValue *operand = NULL;

    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        operands = arena_alloc_array(params->arena, MlirValue*, 1);
        operands[0] = operand;
        n_operands = 1;
    }

    // Parse any attributes and result types using helper functions
    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirType **result_types = NULL;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, NULL);

    // Infer result type if not already parsed from trailing type
    if (n_result_types == 0) {
        MlirType *operand_type = operand ? mlir_value_get_type(operand) : NULL;
        MlirType *result_type = NULL;
        if (operand_type) {
            string operand_type_str = mlir_type_to_string(params->arena, operand_type);
            if (str_eq(operand_type_str, str_lit("!tt.ptr<f32>"))) {
                result_type = mlir_type_create_from_string(params->arena, str_lit("tensor<16x!tt.ptr<f32>>"));
            } else if (str_eq(operand_type_str, str_lit("i32"))) {
                result_type = mlir_type_create_from_string(params->arena, str_lit("tensor<16xi32>"));
            } else {
                result_type = mlir_type_create_from_string(params->arena, str_lit("tensor<16xi32>"));
            }
        } else {
            result_type = mlir_type_create_from_string(params->arena, str_lit("tensor<16xi32>"));
        }
        result_types = arena_alloc_array(params->arena, MlirType*, 1);
        result_types[0] = result_type;
        n_result_types = 1;
    }

    // Parse optional location
    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // Create the operation at the end
    MlirOperation *op = mlir_op_create(params->arena, params->op_type, str_lit(""),
                                      attributes, n_attributes,
                                      result_types, n_result_types,
                                      params->lhs_results, params->n_lhs_results,
                                      operands, n_operands,
                                      NULL, 0,
                                      op_location, params->unnumbered_loc_def,
                                      str_lit(""), params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_tt_make_range_op(Parser *parser, const OperationParserParams *params) {
    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    bool have_end = false;
    int64_t end_val = 0;
    bool have_start = false;
    int64_t start_val = 0;

    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);

        if (parser_peek(parser, TK_NAME)) {
            parser_expect(parser, TK_NAME); // "end"
            parser_expect(parser, TK_EQUAL);
            if (parser_peek(parser, TK_INTEGER)) {
                string ival = parser_token_str(parser);
                char *buffer = arena_alloc_array(parser->arena, char, ival.size + 1);
                memcpy(buffer, ival.str, ival.size);
                buffer[ival.size] = '\0';
                end_val = strtoll(buffer, NULL, 10);
                parser_expect(parser, TK_INTEGER);
            }
            if (parser_peek(parser, TK_COLON)) {
                parser_expect(parser, TK_COLON);
                if (parser_peek(parser, TK_NAME)) parser_expect(parser, TK_NAME);
            }
            have_end = true;
        }

        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);

        if (parser_peek(parser, TK_NAME)) {
            parser_expect(parser, TK_NAME); // "start"
            parser_expect(parser, TK_EQUAL);
            if (parser_peek(parser, TK_INTEGER)) {
                string ival = parser_token_str(parser);
                char *buffer = arena_alloc_array(parser->arena, char, ival.size + 1);
                memcpy(buffer, ival.str, ival.size);
                buffer[ival.size] = '\0';
                start_val = strtoll(buffer, NULL, 10);
                parser_expect(parser, TK_INTEGER);
            }
            if (parser_peek(parser, TK_COLON)) {
                parser_expect(parser, TK_COLON);
                if (parser_peek(parser, TK_NAME)) parser_expect(parser, TK_NAME);
            }
            have_start = true;
        }

        parser_expect(parser, TK_RBRACE);
    }

    if (have_end) {
        append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                    create_integer_attr(parser, str_lit("end"), end_val));
    }
    if (have_start) {
        append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                    create_integer_attr(parser, str_lit("start"), start_val));
    }

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirType **result_types = NULL;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, NULL);

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        NULL,
        0,
        NULL,
        0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_tt_addptr_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);

    while (parser_peek(parser, TK_REGISTER)) {
        if (!parse_register_operand(parser, &operands, true)) {
            OperationParserResult empty = {0};
            return empty;
        }
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        else break;
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirType **result_types = NULL;
    size_t n_result_types = 0;

    while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);

        string pointer_sig = str_lit("");
        if (parse_type_string(parser, &pointer_sig)) {
            result_types = arena_alloc_array(params->arena, MlirType*, 1);
            result_types[0] = mlir_type_create_from_string(params->arena, pointer_sig);
            n_result_types = 1;
        }

        while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);
        if (parser_peek(parser, TK_COMMA)) {
            do {
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            } while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE));
        }
    } else if (operands.size > 0) {
        MlirValue *first = operands.data[0];
        MlirType *first_type = first ? mlir_value_get_type(first) : NULL;
        if (first_type) {
            result_types = arena_alloc_array(params->arena, MlirType*, 1);
            result_types[0] = first_type;
            n_result_types = 1;
        }
    }

    while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) op_location = params->unnumbered_loc_def;

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        NULL, 0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    MlirValue **results = NULL;
    size_t n_results = 0;
    results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_tensor_extract_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    // Parse the tensor operand to extract from
    if (parser_peek(parser, TK_REGISTER)) {
        string tensor_reg = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *tensor_val = symbol_table_lookup(&parser->symbol_table, tensor_reg);
        if (!tensor_val) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, tensor_val);
    } else {
        parser_error(parser, str_lit("Expected tensor operand"), parser->first, parser->last);
        OperationParserResult empty = {0};
        return empty;
    }

    // Parse indices in brackets if present
    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string idx_reg = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                while (parser_peek(parser, TK_HASH_NAME)) {
                    consume_optional_hash_selector(parser);
                }
                MlirValue *idx_val = symbol_table_lookup(&parser->symbol_table, idx_reg);
                if (!idx_val) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    OperationParserResult empty = {0};
                    return empty;
                }
                VecValue_push_back(parser->arena, &operands, idx_val);
                if (parser_peek(parser, TK_COMMA)) {
                    parser_expect(parser, TK_COMMA);
                }
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirType **result_types = NULL;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, NULL);

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        NULL,
        0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_memref_load_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    if (parser_peek(parser, TK_REGISTER)) {
        string reg = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *val = lookup_or_create_value(parser, reg, str_lit("unknown"));
        VecValue_push_back(parser->arena, &operands, val);
    }

    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        bool first = true;
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (!first && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            first = false;
            if (parser_peek(parser, TK_REGISTER)) {
                string idx = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *val = lookup_or_create_value(parser, idx, str_lit("index"));
                VecValue_push_back(parser->arena, &operands, val);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;
    MlirType **result_types = NULL;
    size_t n_result_types = 0;

    MlirValue **results = params->lhs_results;
    size_t n_results = params->n_lhs_results;

    // Parse attributes from both <{...}> and {...} blocks
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    // Parse result types (memref.load operations can have result types)
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, NULL);

    // Parse optional location
    MlirLocation *op_location = parse_optional_location(parser);

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        params->opname,
        attributes, n_attributes,
        result_types, n_result_types,
        results, n_results,
        operands.data, operands.size,
        NULL, 0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_memref_store_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    // Parse operands: store value, memref, [indices...]
    if (parser_peek(parser, TK_REGISTER)) {
        string reg = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *val = lookup_or_create_value(parser, reg, str_lit("unknown"));
        VecValue_push_back(parser->arena, &operands, val);
    }
    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
    if (parser_peek(parser, TK_REGISTER)) {
        string reg = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *val = lookup_or_create_value(parser, reg, str_lit("unknown"));
        VecValue_push_back(parser->arena, &operands, val);
    }

    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        bool first = true;
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (!first && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            first = false;
            if (parser_peek(parser, TK_REGISTER)) {
                string idx = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *val = lookup_or_create_value(parser, idx, str_lit("index"));
                VecValue_push_back(parser->arena, &operands, val);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;
    MlirType **result_types = NULL;
    size_t n_result_types = 0;

    // memref.store operations do not have results
    MlirValue **results = NULL;
    size_t n_results = 0;

    // Parse attributes from both <{...}> and {...} blocks
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    // Parse type information and location after colon (memref.store doesn't have result types)
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            if (angle == 0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            parser_next_token(parser);
        }
    }

    // Parse optional location
    MlirLocation *op_location = parse_optional_location(parser);

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        params->opname,
        attributes, n_attributes,
        result_types, n_result_types,
        results, n_results,
        operands.data, operands.size,
        NULL, 0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_vector_print_op(Parser *parser, const OperationParserParams *params) {
    // Parse operands inside parentheses or as bare register(s)
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        bool first = true;
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (!first && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            first = false;
            if (parser_peek(parser, TK_REGISTER)) {
                string reg = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *val = lookup_or_create_value(parser, reg, str_lit("unknown"));
                VecValue_push_back(parser->arena, &operands, val);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    } else {
        while (parser_peek(parser, TK_REGISTER)) {
            string reg = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            MlirValue *val = lookup_or_create_value(parser, reg, str_lit("unknown"));
            VecValue_push_back(parser->arena, &operands, val);
            if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            else break;
        }
    }

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            if (angle == 0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            parser_next_token(parser);
        }
    }

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        NULL,
        0,
        NULL,
        0,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        NULL,
        0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = NULL,
        .n_results = 0,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_std_constant_op(Parser *parser, const OperationParserParams *params) {
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            parser_next_token(parser);
        }
        parser_expect(parser, TK_RPAREN);
    }

    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        int depth = 1;
        while (depth > 0 && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_LBRACE)) depth++;
            else if (parser_peek(parser, TK_RBRACE)) depth--;
            parser_next_token(parser);
        }
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirType **result_types = NULL;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity,
                      params->op_type, NULL);

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        NULL,
        0,
        NULL,
        0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_tt_reduce_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 1);

    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!operand) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    OperationParserResult empty = {0};
                    return empty;
                }
                VecValue_push_back(parser->arena, &operands, operand);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    if (parser_peek(parser, TK_LANGLE)) {
        parser_expect(parser, TK_LANGLE);
        if (parser_peek(parser, TK_LBRACE)) {
            parser_expect(parser, TK_LBRACE);
            while (!parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_NAME)) {
                    string attr_name = parser_token_str(parser);
                    parser_expect(parser, TK_NAME);
                    if (parser_peek(parser, TK_EQUAL)) {
                        parser_expect(parser, TK_EQUAL);
                        if (parser_peek(parser, TK_INTEGER)) {
                            string int_val = parser_token_str(parser);
                            parser_expect(parser, TK_INTEGER);
                            int64_t val = 0;
                            for (size_t i = 0; i < int_val.size; i++) {
                                char c = int_val.str[i];
                                if (c >= '0' && c <= '9') val = val * 10 + (c - '0');
                            }
                            append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                                        create_integer_attr(parser, attr_name, val));

                            if (parser_peek(parser, TK_COLON)) {
                                parser_expect(parser, TK_COLON);
                                if (parser_peek(parser, TK_NAME)) parser_expect(parser, TK_NAME);
                            }
                        }
                    }
                } else {
                    parser_next_token(parser);
                }
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            }
            parser_expect(parser, TK_RBRACE);
        }
        if (parser_peek(parser, TK_RANGLE)) parser_expect(parser, TK_RANGLE);
    }

    MlirRegion **regions = NULL;
    size_t n_regions = 0;
    if (parser_peek(parser, TK_LPAREN_BRACE) || parser_peek(parser, TK_LPAREN)) {
        if (parser_peek(parser, TK_LPAREN_BRACE)) {
            parser_expect(parser, TK_LPAREN_BRACE);
        } else {
            parser_expect(parser, TK_LPAREN);
        }
        MlirRegion *region = NULL;
        if (parser_peek(parser, TK_LBRACE_END)) {
            region = parse_region(parser);
        }
        parser_expect(parser, TK_RPAREN);
        if (region) {
            regions = arena_alloc_array(params->arena, MlirRegion*, 1);
            regions[0] = region;
            n_regions = 1;
        }
    }

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            int depth = 0;
            while (!parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LPAREN)) depth++;
                else if (parser_peek(parser, TK_RPAREN)) {
                    if (depth == 0) break;
                    depth--;
                }
                parser_next_token(parser);
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    MlirType **result_types = NULL;
    size_t n_result_types = 0;
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        string type_str = str_lit("");
        if (parse_type_string(parser, &type_str)) {
            result_types = arena_alloc_array(params->arena, MlirType*, 1);
            result_types[0] = mlir_type_create_from_string(params->arena, type_str);
            n_result_types = 1;
        }
    }

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    if (n_result_types == 0) {
        parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity,
                          params->op_type, NULL);
    }

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        regions,
        n_regions,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_cf_br_op(Parser *parser, const OperationParserParams *params) {
    // Parse cf.br ^bbX(%args : types)
    string target = str_lit("^bb1");
    if (parser_peek(parser, TK_CARET_NAME)) {
        target = parser_token_str(parser);
        parser_expect(parser, TK_CARET_NAME);
    }

    VecValue branch_args;
    VecValue_reserve(parser->arena, &branch_args, 4);
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string vr = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *v = symbol_table_lookup(&parser->symbol_table, vr);
                if (v) VecValue_push_back(parser->arena, &branch_args, v);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else if (parser_peek(parser, TK_COLON)) {
                parser_expect(parser, TK_COLON);
                int angle = 0;
                while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                    if (parser_peek(parser, TK_LANGLE)) angle++;
                    else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
                    parser_next_token(parser);
                }
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;
    append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                create_string_attr(parser, str_lit("_target"), target));

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        NULL,
        0,
        params->lhs_results,
        params->n_lhs_results,
        branch_args.data,
        branch_args.size,
        NULL,
        0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = NULL,
        .n_results = 0,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_cf_cond_br_op(Parser *parser, const OperationParserParams *params) {
    // Parse: cf.cond_br %cond, ^bbX[(args : types)], ^bbY[(args : types)] [loc]
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 1);

    if (parser_peek(parser, TK_REGISTER)) {
        string reg = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *cond = symbol_table_lookup(&parser->symbol_table, reg);
        if (cond) VecValue_push_back(parser->arena, &operands, cond);
    }

    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);

    string ttrue = str_lit("");
    string tfalse = str_lit("");
    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    if (parser_peek(parser, TK_CARET_NAME)) {
        ttrue = parser_token_str(parser);
        parser_expect(parser, TK_CARET_NAME);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            int64_t ntrue = 0;
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string vr = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);
                    MlirValue *v = symbol_table_lookup(&parser->symbol_table, vr);
                    if (v) VecValue_push_back(parser->arena, &operands, v);
                    ntrue++;
                    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
                } else if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    int angle = 0;
                    while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                        if (parser_peek(parser, TK_LANGLE)) angle++;
                        else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
                        parser_next_token(parser);
                    }
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RPAREN);
            append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                        create_integer_attr(parser, str_lit("_ntrue"), ntrue));
        }
    }

    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);

    if (parser_peek(parser, TK_CARET_NAME)) {
        tfalse = parser_token_str(parser);
        parser_expect(parser, TK_CARET_NAME);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            int64_t nfalse = 0;
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string vr = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);
                    MlirValue *v = symbol_table_lookup(&parser->symbol_table, vr);
                    if (v) VecValue_push_back(parser->arena, &operands, v);
                    nfalse++;
                    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
                } else if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    int angle = 0;
                    while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                        if (parser_peek(parser, TK_LANGLE)) angle++;
                        else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
                        parser_next_token(parser);
                    }
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RPAREN);
            append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                        create_integer_attr(parser, str_lit("_nfalse"), nfalse));
        }
    }

    append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                create_string_attr(parser, str_lit("_true"), ttrue));
    append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                create_string_attr(parser, str_lit("_false"), tfalse));

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        NULL,
        0,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        NULL,
        0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = NULL,
        .n_results = 0,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_linalg_fill_op(Parser *parser, const OperationParserParams *params) {
    // Parse linalg.fill ins(%val : type) outs(%tensor : type)

    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("ins"))) {
        parser_expect(parser, TK_NAME);
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                parser_expect(parser, TK_REGISTER);
                if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                        parser_next_token(parser);
                    }
                }
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("outs"))) {
        parser_expect(parser, TK_NAME);
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                parser_expect(parser, TK_REGISTER);
                if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                        parser_next_token(parser);
                    }
                }
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        NULL,
        0,
        NULL,
        0,
        params->lhs_results,
        params->n_lhs_results,
        NULL,
        0,
        NULL,
        0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = NULL,
        .n_results = 0,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_affine_load_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);

    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *memref_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!memref_operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, memref_operand);
    }

    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *idx = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!idx) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    OperationParserResult empty = {0};
                    return empty;
                }
                VecValue_push_back(parser->arena, &operands, idx);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirType **result_types = NULL;
    size_t n_result_types = 0;

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("memref"))) {
            parser_expect(parser, TK_NAME);
            if (parser_peek(parser, TK_LANGLE)) {
                parser_expect(parser, TK_LANGLE);
                string element_type = str_lit("unknown");

                if (parser_peek(parser, TK_TYPE_DIM)) {
                    string type_dim = parser_token_str(parser);
                    parser_expect(parser, TK_TYPE_DIM);
                    size_t last_x = SIZE_MAX;
                    for (size_t i = 0; i < type_dim.size; i++) {
                        if (type_dim.str[i] == 'x') last_x = i;
                    }
                    if (last_x != SIZE_MAX && last_x + 1 < type_dim.size) {
                        element_type = str_substr(type_dim, last_x + 1, type_dim.size - last_x - 1);
                    }
                }

                if (parser_peek(parser, TK_NAME)) {
                    element_type = parser_token_str(parser);
                    parser_expect(parser, TK_NAME);
                }

                result_types = arena_alloc_array(params->arena, MlirType*, 1);
                result_types[0] = mlir_type_create_from_string(params->arena, element_type);
                n_result_types = 1;

                if (parser_peek(parser, TK_RANGLE)) parser_expect(parser, TK_RANGLE);
            }
        }
    }

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    if (n_result_types == 0) {
        parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity,
                          params->op_type, NULL);
    }

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        NULL,
        0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_index_constant_op(Parser *parser, const OperationParserParams *params) {
    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    MlirType **result_types = NULL;
    size_t n_result_types = 0;

    // Parse the constant value if present
    if (parser_peek(parser, TK_INTEGER)) {
        parser_expect(parser, TK_INTEGER);
    } else {
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
               !parser_peek(parser, TK_NAME) && !parser_peek(parser, TK_COLON)) {
            parser_next_token(parser);
        }
    }

    // Capture any additional attributes and result type information
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, NULL);

    if (n_result_types == 0) {
        result_types = arena_alloc_array(params->arena, MlirType*, 1);
        result_types[0] = mlir_type_create_from_string(params->arena, str_lit("index"));
        n_result_types = 1;
    }

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        NULL,
        0,
        NULL,
        0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_tensor_splat_op(Parser *parser, const OperationParserParams *params) {
    // Parse tensor.splat %value[%dim1, %dim2] : tensor<type>
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 3);

    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *value_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!value_operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, value_operand);
    }

    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *dim = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!dim) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    OperationParserResult empty = {0};
                    return empty;
                }
                VecValue_push_back(parser->arena, &operands, dim);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;

    MlirType **result_types = NULL;
    size_t n_result_types = 0;

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("tensor"))) {
            parser_expect(parser, TK_NAME);
            if (parser_peek(parser, TK_LANGLE)) {
                parser_expect(parser, TK_LANGLE);
                string tensor_type = str_lit("tensor<");
                int depth = 1;
                while (depth > 0 && !parser_peek(parser, TK_EOF)) {
                    string token_str = parser_token_str(parser);
                    if (parser_peek(parser, TK_LANGLE)) depth++;
                    else if (parser_peek(parser, TK_RANGLE)) depth--;
                    tensor_type = str_concat(parser->arena, tensor_type, token_str);
                    parser_next_token(parser);
                }

                result_types = arena_alloc_array(params->arena, MlirType*, 1);
                result_types[0] = mlir_type_create_from_string(params->arena, tensor_type);
                n_result_types = 1;
            }
        }
    }

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        NULL,
        0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_arith_select_op(Parser *parser, const OperationParserParams *params) {
    // Parse arith.select %cond, %true_val, %false_val : cond_type, val_type
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 3);

    // Parse three operands
    for (int i = 0; i < 3; i++) {
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            MlirValue *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                OperationParserResult empty = {0};
                return empty;
            }
            VecValue_push_back(parser->arena, &operands, operand);
            if (i < 2 && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        } else {
            parser_error(parser, str_lit("Expected register operand"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
    }

    // Parse any additional attributes
    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirType **result_types = NULL;
    size_t n_result_types = 0;

    // Parse ": type1, type2" - result type is type2 (the value type)
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        // Skip first type (condition type)
        while (!parser_peek(parser, TK_COMMA) && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE)) {
            parser_next_token(parser);
        }

        if (parser_peek(parser, TK_COMMA)) {
            parser_expect(parser, TK_COMMA);

            // Parse the second type (result type)
            string result_type = str_lit("");
            while (!parser_peek(parser, TK_EOF) &&
                   !parser_peek(parser, TK_NEWLINE) &&
                   !(parser_peek(parser, TK_NAME) &&
                     str_eq(parser_token_str(parser), str_lit("loc")))) {
                if (result_type.size > 0) {
                    result_type = str_concat(parser->arena, result_type, parser_token_str(parser));
                } else {
                    result_type = parser_token_str(parser);
                }
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            }

            if (result_type.size > 0) {
                result_types = arena_alloc_array(params->arena, MlirType*, 1);
                result_types[0] = mlir_type_create_from_string(params->arena, result_type);
                n_result_types = 1;
            }
        }
    }

    // Parse any remaining result types if not already parsed
    if (n_result_types == 0) {
        parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, NULL);
    }

    // Parse optional location
    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // Create the operation at the end
    MlirOperation *op = mlir_op_create(params->arena, params->op_type, str_lit(""),
                                      attributes, n_attributes,
                                      result_types, n_result_types,
                                      params->lhs_results, params->n_lhs_results,
                                      operands.data, operands.size,
                                      NULL, 0,
                                      op_location, params->unnumbered_loc_def,
                                      str_lit(""), params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_tt_call_op(Parser *parser, const OperationParserParams *params) {
    // Parse tt.call @function(%args) : (arg_types) -> result_type

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    if (parser_peek(parser, TK_FUNCTION_NAME)) {
        string fname = parser_token_str(parser);
        parser_expect(parser, TK_FUNCTION_NAME);
        append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                    create_string_attr(parser, str_lit("callee"), fname));
    }

    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!operand) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    OperationParserResult empty = {0};
                    return empty;
                }
                VecValue_push_back(parser->arena, &operands, operand);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    MlirType **result_types = NULL;
    size_t n_result_types = 0;

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        if (parser_peek(parser, TK_LPAREN)) {
            int depth = 0;
            do {
                if (parser_peek(parser, TK_LPAREN)) depth++;
                else if (parser_peek(parser, TK_RPAREN)) depth--;
                parser_next_token(parser);
            } while (depth > 0 && !parser_peek(parser, TK_EOF));
        }

        if (parser_peek(parser, TK_ARROW)) {
            parser_expect(parser, TK_ARROW);

            string result_type = str_lit("");
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
                   !(parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc")))) {
                string tok = parser_token_str(parser);
                result_type = result_type.size ? str_concat(parser->arena, result_type, tok) : tok;
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            }

            if (result_type.size > 0) {
                result_types = arena_alloc_array(params->arena, MlirType*, 1);
                result_types[0] = mlir_type_create_from_string(params->arena, result_type);
                n_result_types = 1;
            }
        }
    }

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        NULL,
        0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_tensor_collapse_shape_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 1);

    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *tensor_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!tensor_operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, tensor_operand);
    }

    if (parser_peek(parser, TK_LBRACKET)) {
        int depth = 0;
        do {
            if (parser_peek(parser, TK_LBRACKET)) depth++;
            else if (parser_peek(parser, TK_RBRACKET)) depth--;
            parser_next_token(parser);
        } while (depth > 0 && !parser_peek(parser, TK_EOF));
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    MlirType **result_types = NULL;
    size_t n_result_types = 0;

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
               (!parser_peek(parser, TK_NAME) || !str_eq(parser_token_str(parser), str_lit("into")))) {
            parser_next_token(parser);
        }

        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("into"))) {
            parser_expect(parser, TK_NAME);

            string result_type = str_lit("");
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
                   !(parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc")))) {
                string tok = parser_token_str(parser);
                result_type = result_type.size ? str_concat(parser->arena, result_type, tok) : tok;
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            }

            if (result_type.size > 0) {
                result_types = arena_alloc_array(params->arena, MlirType*, 1);
                result_types[0] = mlir_type_create_from_string(params->arena, result_type);
                n_result_types = 1;
            }
        }
    }

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    if (n_result_types == 0) {
        parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity,
                          params->op_type, NULL);
    }

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        NULL,
        0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

static void parse_generic_operation_impl(Parser *parser, MlirOperation *op) {
    // Remember the start of this operation's source line for accurate comment capture
    {
        int64_t pos = (int64_t)parser->first;
        while (pos > 0) {
            unsigned char c = parser->input[pos - 1];
            if (c == '\n' || c == '\r') break;
            pos--;
        }
        mlir_operation_set_source_line_start(op, pos);
    }
    // Skip non-structural tokens (e.g., cmp predicates) until operands/attrs/types
    while (!parser_peek(parser, TK_EOF) &&
           !(parser_peek(parser, TK_REGISTER) || parser_peek(parser, TK_LBRACKET) ||
             parser_peek(parser, TK_COLON) || parser_peek(parser, TK_LBRACE) ||
             parser_peek(parser, TK_LBRACE_END) || parser_peek(parser, TK_NEWLINE) ||
             parser_peek(parser, TK_LPAREN_BRACE))) {
        parser_next_token(parser);
    }

    // Parse operands until attributes, result type, or region begin
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    while ((parser_peek(parser, TK_REGISTER) || parser_peek(parser, TK_LBRACKET)) &&
           !parser_peek(parser, TK_COLON) && !parser_peek(parser, TK_LBRACE)) {
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);

            MlirValue *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                return;
            }
            VecValue_push_back(parser->arena, &operands, operand);
        } else if (parser_peek(parser, TK_LBRACKET)) {
            // Skip indexing syntax like [%arg1]
            parser_expect(parser, TK_LBRACKET);
            while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string reg_str2 = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);
                    MlirValue *operand2 = symbol_table_lookup(&parser->symbol_table, reg_str2);
                    if (!operand2) {
                        parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                        return;
                    }
                    VecValue_push_back(parser->arena, &operands, operand2);
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RBRACKET);
        }

        if (parser_peek(parser, TK_COMMA)) {
            parser_expect(parser, TK_COMMA);
        } else {
            break;
        }
    }

    set_op_operands(op, operands.data, operands.size);

    // Attributes and result types are parsed generically later
    // Handle generic attributes and result types before scanning for regions
    parse_generic_attrs_and_result_type(parser, op);
    // Capture trailing loc() immediately if present to avoid getting skipped
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        mlir_operation_set_location(op, parse_loc(parser));
    }

    // Parse regions (if any), for now we assume 0 or 1 regions
    while (!(parser_peek(parser, TK_LBRACE_END)
             || parser_peek(parser, TK_NEWLINE)
             || parser_peek(parser, TK_LPAREN_BRACE))) {
        parser_next_token(parser);
    }
    bool lparen_brace = false;
    if (parser_peek(parser, TK_LPAREN_BRACE)) {
        lparen_brace = true;
        parser_next_token(parser);
    }
    if (parser_peek(parser, TK_LBRACE_END)) {
        // Before consuming a region, parse any trailing attributes/result types
        parse_generic_attrs_and_result_type(parser, op);
        MlirRegion *region = parse_region(parser);
        mlir_op_add_region(parser->arena, op, region);
        // After region, there can be another signature/attrs like ": (tys) -> ty"
        parse_generic_attrs_and_result_type(parser, op);
    }
    if (lparen_brace) {
        parser_expect(parser, TK_RPAREN);
        while (!parser_peek(parser, TK_NEWLINE)) {
            parser_next_token(parser);
        }
    }
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        mlir_operation_set_location(op, parse_loc(parser));
    }
}
OperationParserResult parse_generic_op(Parser *parser, const OperationParserParams *params) {
    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        NULL, 0,
        NULL, 0,
        params->lhs_results,
        params->n_lhs_results,
        NULL, 0,
        NULL, 0,
        NULL,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    if (params->op_type == OP_TYPE_UNREGISTERED && params->opname.size > 0) {
        mlir_operation_set_name(op, params->opname.str, params->opname.size);
    }

    parse_generic_operation_impl(parser, op);

    MlirLocation *op_location = mlir_operation_get_location(op);
    if (!op_location) op_location = params->unnumbered_loc_def;
    if (op_location) mlir_operation_set_location(op, op_location);

    size_t n_result_types = mlir_operation_num_result_types(op);
    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, NULL, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}



OperationParserResult parse_tt_func_op(Parser *parser, const OperationParserParams *params) {
    // Record source line start - will be set on the operation created at the end
    // Parse tt.func public @function_name(%arg0: type, %arg1: type, ...)

    // Initialize variables that will be used at the end
    MlirLocation *op_location = NULL;
    MlirRegion *func_region = NULL;
    MlirValue **func_operands = NULL;
    size_t n_func_operands = 0;

    // Vector to collect function arguments as we parse them
    VecValue func_args;
    VecValue_reserve(parser->arena, &func_args, 8);

    // Capture visibility keyword if present
    string visibility = str_lit("private");  // default
    if (parser_peek(parser, TK_NAME) && (str_eq(parser_token_str(parser), str_lit("public")) || str_eq(parser_token_str(parser), str_lit("private")))) {
        visibility = parser_token_str(parser);
        parser_expect(parser, TK_NAME);
    }

    MlirAttribute **attrs = NULL; size_t n_attrs = 0, cap_attrs = 0;
    append_attr(parser, &attrs, &n_attrs, &cap_attrs, create_string_attr(parser, str_lit("visibility"), visibility));

    // Parse @function_name
    if (parser_peek(parser, TK_FUNCTION_NAME)) {
        // Capture function symbol name into an attribute for printing
        string fname_with_at = parser_token_str(parser);
        parser_expect(parser, TK_FUNCTION_NAME);

        // Remove the '@' prefix to get just the function name
        string fname = str_substr(fname_with_at, 1, fname_with_at.size - 1);

        // Store as string attribute 'sym_name'
        append_attr(parser, &attrs, &n_attrs, &cap_attrs, create_string_attr(parser, str_lit("sym_name"), fname));
    } else {
        // Try to skip tokens until we find a parenthesis
        while (!parser_peek(parser, TK_LPAREN) && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
            parser_next_token(parser);
        }
    }

    // Parse function arguments
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);

        // Arguments will be collected in the func_args vector

        // Accumulate a textual params signature when args are type-only (declarations)
        string params_sig = str_lit("");

        // Parse arguments directly in one pass
        int paren_depth = 0;
        while (!parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_LPAREN)) {
                paren_depth++;
                parser_next_token(parser);
            } else if (parser_peek(parser, TK_RPAREN)) {
                if (paren_depth == 0) {
                    // Found the main closing paren
                    break;
                }
                paren_depth--;
                parser_next_token(parser);
            } else if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);

                // Parse colon and type
                if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);

                    MlirValue *arg = mlir_value_create(parser->arena, BLOCK_ARG);
                    mlir_value_set_register_name(arg, string_data_or_null(reg_str), reg_str.size);
                    mlir_value_set_result_index(arg, (uint32_t)func_args.size);
                    MlirType *arg_type = NULL;

                    // Parse the argument type
                    if (parser_peek(parser, TK_EXCLAMATION)) {
                        parser_expect(parser, TK_EXCLAMATION);

                        if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                            string type_name = parser_token_str(parser);
                            if (parser_peek(parser, TK_NAME)) {
                                parser_expect(parser, TK_NAME);
                            } else {
                                parser_expect(parser, TK_NAME_DOT_NAME);
                            }

                            // Handle complex types like !tt.ptr<f32, 1>
                            if (str_eq(type_name, str_lit("tt.ptr"))) {
                                if (parser_peek(parser, TK_LANGLE)) {
                                    parser_expect(parser, TK_LANGLE);

                                    // Parse everything inside angle brackets
                                    string type_content = str_lit("");
                                    while (!parser_peek(parser, TK_RANGLE) && !parser_peek(parser, TK_EOF)) {
                                        string token_str = parser_token_str(parser);
                                        type_content = str_concat(parser->arena, type_content, token_str);
                                        parser_next_token(parser);
                                    }

                                    if (parser_peek(parser, TK_RANGLE)) {
                                        parser_expect(parser, TK_RANGLE);
                                        string full_type_str = str_concat(parser->arena, str_lit("!tt.ptr<"), str_concat(parser->arena, type_content, str_lit(">")));
                                        arg_type = mlir_type_create_from_string(parser->arena, full_type_str);
                                    }
                                } else {
                                    arg_type = mlir_type_create_from_string(parser->arena, str_lit("!tt.ptr"));
                                }
                            } else {
                                arg_type = mlir_type_create_from_string(parser->arena, str_concat(parser->arena, str_lit("!"), type_name));
                            }
                        } else {
                            arg_type = mlir_type_create_from_string(parser->arena, str_lit("!unknown"));
                        }
                    } else if (parser_peek(parser, TK_NAME)) {
                        // Simple type like i32 or complex type like tensor<2x256xf32>
                        string type_name = parser_token_str(parser);
                        parser_expect(parser, TK_NAME);

                        // Handle complex types with angle brackets (e.g., tensor<2x256xf32>)
                        if (parser_peek(parser, TK_LANGLE)) {
                            parser_expect(parser, TK_LANGLE);
                            string angle_content = str_lit("");
                            int bracket_depth = 1;
                            while (bracket_depth > 0 && !parser_peek(parser, TK_EOF)) {
                                if (parser_peek(parser, TK_LANGLE)) {
                                    bracket_depth++;
                                } else if (parser_peek(parser, TK_RANGLE)) {
                                    bracket_depth--;
                                    if (bracket_depth == 0) {
                                        parser_expect(parser, TK_RANGLE);
                                        break;
                                    }
                                }
                                string token_str = parser_token_str(parser);
                                angle_content = str_concat(parser->arena, angle_content, token_str);
                                parser_next_token(parser);
                            }
                            // Reconstruct the full type string
                            string full_type = str_concat(parser->arena, type_name, str_concat(parser->arena, str_lit("<"), str_concat(parser->arena, angle_content, str_lit(">"))));
                            arg_type = mlir_type_create_from_string(parser->arena, full_type);
                        } else {
                            arg_type = mlir_type_create_from_string(parser->arena, type_name);
                        }
                    } else {
                        arg_type = mlir_type_create_from_string(parser->arena, str_lit("unknown"));
                    }

                    // Optional per-arg attribute dict: { ... }
                    if (parser_peek(parser, TK_LBRACE)) {
                        int depth = 0;
                        // Try to detect simple 'tt.divisibility = <int> : <type>' pattern
                        // without constructing a full attribute AST.
                        // Consume '{'
                        parser_expect(parser, TK_LBRACE);
                        depth++;
                        // Read tokens until matching '}'
                        while (depth > 0 && !parser_peek(parser, TK_EOF)) {
                            if (parser_peek(parser, TK_LBRACE)) { depth++; parser_next_token(parser); continue; }
                            if (parser_peek(parser, TK_RBRACE)) { depth--; parser_next_token(parser); if (depth==0) break; else continue; }
                            // Match name possibly with dots
                            if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                                string name = parser_token_str(parser);
                                parser_next_token(parser);
                                if (str_eq(name, str_lit("tt.divisibility"))) {
                                    // Expect '=' integer ':' type
                                    if (parser_peek(parser, TK_EQUAL)) parser_expect(parser, TK_EQUAL);
                                    if (parser_peek(parser, TK_INTEGER)) {
                                        string ival = parser_token_str(parser);
                                        // parse integer
                                        int64_t v = 0; for (size_t k=0;k<ival.size;k++){ char c=ival.str[k]; if (c>='0' && c<='9') v = v*10 + (c-'0'); }
                                        parser_expect(parser, TK_INTEGER);
                                        // optional ':' type
                                        MlirType *dtype = NULL;
                                        if (parser_peek(parser, TK_COLON)) {
                                            parser_expect(parser, TK_COLON);
                                            string tstr = str_lit("");
                                            // Reuse type parsing helper from mlir_parser.c via forward decl
                                            // Here, mimic parse_type_string: collect until '}' or ',' or 'loc'
                                            int angle = 0;
                                            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_COMMA)) {
                                                if (parser_peek(parser, TK_LANGLE)) angle++;
                                                else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--;
                                                if (angle==0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
                                                string tok = parser_token_str(parser);
                                                tstr = tstr.size ? str_concat(parser->arena, tstr, tok) : tok;
                                                parser_next_token(parser);
                                                if (angle==0 && (parser_peek(parser, TK_RBRACE) || parser_peek(parser, TK_COMMA))) break;
                                            }
                                            dtype = mlir_type_create_from_string(parser->arena, tstr);
                                        }
                                        MlirType *dtype_actual = dtype ? dtype : mlir_type_create_from_string(parser->arena, str_lit("i32"));
                                        mlir_value_set_divisibility(arg, true, v, dtype_actual);
                                    }
                                } else if (str_eq(name, str_lit("tt.max_divisibility"))) {
                                    // Expect '=' integer ':' type
                                    if (parser_peek(parser, TK_EQUAL)) parser_expect(parser, TK_EQUAL);
                                    if (parser_peek(parser, TK_INTEGER)) {
                                        string ival = parser_token_str(parser);
                                        // parse integer
                                        int64_t v = 0; for (size_t k=0;k<ival.size;k++){ char c=ival.str[k]; if (c>='0' && c<='9') v = v*10 + (c-'0'); }
                                        parser_expect(parser, TK_INTEGER);
                                        // optional ':' type
                                        MlirType *dtype = NULL;
                                        if (parser_peek(parser, TK_COLON)) {
                                            parser_expect(parser, TK_COLON);
                                            string tstr = str_lit("");
                                            int angle = 0;
                                            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_COMMA)) {
                                                if (parser_peek(parser, TK_LANGLE)) angle++;
                                                else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--;
                                                if (angle==0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
                                                string tok = parser_token_str(parser);
                                                tstr = tstr.size ? str_concat(parser->arena, tstr, tok) : tok;
                                                parser_next_token(parser);
                                                if (angle==0 && (parser_peek(parser, TK_RBRACE) || parser_peek(parser, TK_COMMA))) break;
                                            }
                                            dtype = mlir_type_create_from_string(parser->arena, tstr);
                                        }
                                        MlirType *dtype_actual = dtype ? dtype : mlir_type_create_from_string(parser->arena, str_lit("i32"));
                                        mlir_value_set_max_divisibility(arg, true, v, dtype_actual);
                                    }
                                }
                            } else {
                                parser_next_token(parser);
                            }
                        }
                    }

                    // Optional trailing per-arg loc()
                    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                        mlir_value_set_location(arg, parse_loc(parser));
                    }

                    if (arg_type) mlir_value_set_type(arg, arg_type);

                    VecValue_push_back(parser->arena, &func_args, arg);
                }
            } else if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION)) {
                // Type-only param (no SSA name); accumulate its textual form
                string ty = str_lit("");
                if (parse_type_string(parser, &ty)) {
                    if (params_sig.size > 0) params_sig = str_concat(parser->arena, params_sig, str_lit(", "));
                    params_sig = str_concat(parser->arena, params_sig, ty);
                } else {
                    parser_next_token(parser);
                }
            } else {
                parser_next_token(parser);
            }
        }

        // Store operands for later assignment to operation
        if (func_args.size > 0) {
            func_operands = func_args.data;
            n_func_operands = func_args.size;
        }

        if (parser_peek(parser, TK_RPAREN)) {
            parser_expect(parser, TK_RPAREN);
        } else {
        }

        // Save params signature if any
        if (params_sig.size > 0) {
            append_attr(parser, &attrs, &n_attrs, &cap_attrs, create_string_attr(parser, str_lit("params_sig"), params_sig));
        }
    }

    // Optional return signature after '->' (capture conservatively until 'attributes' or body)
    string ret_sig = str_lit("");
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
            if (parser_peek(parser, TK_NAME)) {
                string nm = parser_token_str(parser);
                if (str_eq(nm, str_lit("attributes")) || str_eq(nm, str_lit("loc"))) break;
            }
            ret_sig = str_concat(parser->arena, ret_sig, parser_token_str(parser));
            parser_next_token(parser);
        }
    }

    // Optionally capture function attributes: attributes { ... }
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("attributes"))) {
        parser_expect(parser, TK_NAME); // attributes
        if (parser_peek(parser, TK_LBRACE)) {
            int depth = 0; parser_expect(parser, TK_LBRACE); depth++;
            // Conservatively skip content; generic op attribute parser can be used later if needed
            while (depth>0 && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LBRACE)) { depth++; }
                else if (parser_peek(parser, TK_RBRACE)) { depth--; }
                parser_next_token(parser);
            }
        }
    }

    // If there is a function body, parse it; otherwise leave as a declaration
    if (parser_peek(parser, TK_LBRACE_END)) {
        // Parse the function body region with special handling for function arguments
        parser_expect(parser, TK_LBRACE_END);
        parser_expect(parser, TK_NEWLINE);

        // Push new scope for this region
        symbol_table_push_scope(parser->arena, &parser->symbol_table);

        // Register function arguments in the new scope
        for (size_t i = 0; i < func_args.size; i++) {
            MlirValue *operand = func_args.data[i];
            if (!operand) continue;
            string name = mlir_value_get_register_name(operand);
            if (name.size > 0) {
                symbol_table_add_value(parser->arena, &parser->symbol_table, name, operand);
            }
        }

        // Parse blocks
        VecBlock blocks;
        VecBlock_reserve(parser->arena, &blocks, 8);
        while (!parser_peek(parser, TK_RBRACE)) {
            MlirBlock *block = parse_block(parser);
            VecBlock_push_back(parser->arena, &blocks, block);
        }
        parser_expect(parser, TK_RBRACE);

        // Pop scope when leaving region
        symbol_table_pop_scope(&parser->symbol_table);

        // Create region
        MlirRegion *region = mlir_region_create(parser->arena);
        for (size_t i = 0; i < blocks.size; i++) {
            mlir_region_add_block(parser->arena, region, blocks.data[i]);
        }

        // Store region for later assignment to operation
        func_region = region;

        if (parser_peek(parser, TK_NAME)) {
            if (str_eq(parser_token_str(parser), str_lit("loc"))) {
                op_location = parse_loc(parser);
            }
        }
    } else {
        // Declaration: no body. Optionally consume trailing loc
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
            op_location = parse_loc(parser);
        }
    }

    // Parse any additional attributes and result types using helper functions
    parse_angle_brace_attributes(parser, &attrs, &n_attrs, &cap_attrs);
    parse_brace_attributes(parser, &attrs, &n_attrs, &cap_attrs);

    MlirType **result_types = NULL;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attrs, &n_attrs, &cap_attrs, params->op_type, NULL);

    // Parse optional location if not already set
    if (!op_location) {
        op_location = parse_optional_location(parser);
        if (!op_location) {
            op_location = params->unnumbered_loc_def;
        }
    }

    // Create the operation at the end
    MlirOperation *op = mlir_op_create(params->arena, params->op_type, str_lit(""),
                                      attrs, n_attrs,
                                      result_types, n_result_types,
                                      params->lhs_results, params->n_lhs_results,
                                      func_operands, n_func_operands,
                                      NULL, 0,
                                      op_location, params->unnumbered_loc_def,
                                      str_lit(""), params->source_line_start);

    // Add the region if it exists
    if (func_region) {
        mlir_op_add_region(params->arena, op, func_region);
    }

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

void parse_func_func(Parser *parser, MlirOperation *op) {
    // Capture optional visibility and function name
    string visibility = str_lit("");
    string fname = str_lit("");
    MlirAttribute **attrs = NULL; size_t n_attrs = 0, cap_attrs = 0;
    attr_list_init_from_op(parser, op, &attrs, &n_attrs, &cap_attrs);
    while (!parser_peek(parser, TK_LPAREN) && !parser_peek(parser, TK_EOF)) {
        if (parser_peek(parser, TK_NAME)) {
            string nm = parser_token_str(parser);
            if (str_eq(nm, str_lit("private")) || str_eq(nm, str_lit("public"))) {
                visibility = nm; parser_expect(parser, TK_NAME); continue;
            }
        }
        if (parser_peek(parser, TK_FUNCTION_NAME)) {
            fname = parser_token_str(parser); parser_expect(parser, TK_FUNCTION_NAME); continue;
        }
        parser_next_token(parser);
    }
    // Parse argument list
    VecValue args; VecValue_reserve(parser->arena, &args, 8);
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        bool first = true;
        string params_sig = str_lit("");
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (!first && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            first = false;
            if (parser_peek(parser, TK_REGISTER)) {
                string reg = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                // Optional type annotation
                MlirType *ty = NULL;
                if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    string t = str_lit("");
                    if (parse_type_string(parser, &t)) {
                        ty = mlir_type_create_from_string(parser->arena, t);

                    }
                }
                MlirValue *arg = mlir_value_create(parser->arena, BLOCK_ARG);
                mlir_value_set_register_name(arg, string_data_or_null(reg), reg.size);
                if (ty) mlir_value_set_type(arg, ty);
                VecValue_push_back(parser->arena, &args, arg);
            } else if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION)) {
                // Type-only argument in declaration form
                string t = str_lit("");
                if (parse_type_string(parser, &t)) {
                    if (params_sig.size > 0) params_sig = str_concat(parser->arena, params_sig, str_lit(", "));
                    params_sig = str_concat(parser->arena, params_sig, t);
                } else {
                    parser_next_token(parser);
                }
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
        if (params_sig.size > 0) {
            append_attr(parser, &attrs, &n_attrs, &cap_attrs, create_string_attr(parser, str_lit("params_sig"), params_sig));
        }
    }
    // Optional return type sequence after '->' (capture text conservatively)
    string ret_sig = str_lit("");
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        // Capture until '{' or loc
        while (!(parser_peek(parser, TK_LBRACE_END) || parser_peek(parser, TK_EOF))) {
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            ret_sig = str_concat(parser->arena, ret_sig, parser_token_str(parser));
            parser_next_token(parser);
        }

    }
    // Parse optional body region (definition) or leave as declaration
    if (parser_peek(parser, TK_LBRACE_END)) {
        MlirRegion *region = parse_region(parser);
        mlir_op_add_region(parser->arena, op, region);
    }

    // Store attributes for classic printing
    if (visibility.size > 0) append_attr(parser, &attrs, &n_attrs, &cap_attrs, create_string_attr(parser, str_lit("visibility"), visibility));
    if (fname.size > 0) append_attr(parser, &attrs, &n_attrs, &cap_attrs, create_string_attr(parser, str_lit("sym_name"), fname));
    if (ret_sig.size > 0) append_attr(parser, &attrs, &n_attrs, &cap_attrs, create_string_attr(parser, str_lit("ret"), ret_sig));
    if (n_attrs > 0) set_op_attributes(op, attrs, n_attrs);
}

OperationParserResult parse_scf_if_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 1);

    if (parser_peek(parser, TK_REGISTER)) {
        string cond_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *condition = symbol_table_lookup(&parser->symbol_table, cond_str);
        if (!condition) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, condition);
    } else {
        parser_error(parser, str_lit("Expected condition register for scf.if"), parser->first, parser->last);
        OperationParserResult empty = {0};
        return empty;
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("attributes"))) {
        parser_expect(parser, TK_NAME);
        parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    }

    MlirType **result_types = NULL;
    size_t n_result_types = 0;

    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            size_t cap = 2;
            result_types = arena_alloc_array(params->arena, MlirType*, cap);
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                string type_str = str_lit("");
                if (!parse_type_string(parser, &type_str)) break;
                if (n_result_types >= cap) {
                    cap *= 2;
                    MlirType **expanded = arena_alloc_array(params->arena, MlirType*, cap);
                    for (size_t i = 0; i < n_result_types; i++) expanded[i] = result_types[i];
                    result_types = expanded;
                }
                result_types[n_result_types++] = mlir_type_create_from_string(params->arena, type_str);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            }
            parser_expect(parser, TK_RPAREN);
            if (n_result_types == 0) result_types = NULL;
        } else {
            string type_str = str_lit("");
            if (parse_type_string(parser, &type_str)) {
                result_types = arena_alloc_array(params->arena, MlirType*, 1);
                result_types[0] = mlir_type_create_from_string(params->arena, type_str);
                n_result_types = 1;
            }
        }
    }

    MlirRegion *then_region = parse_region(parser);
    MlirRegion *else_region = NULL;
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("else"))) {
        parser_expect(parser, TK_NAME);
        else_region = parse_region(parser);
    }

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    size_t n_regions = else_region ? 2 : 1;
    MlirRegion **regions = arena_alloc_array(params->arena, MlirRegion*, n_regions);
    regions[0] = then_region;
    if (else_region) regions[1] = else_region;

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        regions,
        n_regions,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_scf_for_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    MlirValue *loop_var = NULL;
    MlirType *iv_type = NULL;

    if (parser_peek(parser, TK_REGISTER)) {
        string loop_var_name = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);

        loop_var = mlir_value_create(parser->arena, BLOCK_ARG);
        mlir_value_set_register_name(loop_var, string_data_or_null(loop_var_name), loop_var_name.size);
        mlir_value_set_type(loop_var, NULL);

        parser_expect(parser, TK_EQUAL);

        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            MlirValue *start_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!start_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                OperationParserResult empty = {0};
                return empty;
            }
            VecValue_push_back(parser->arena, &operands, start_operand);
        }

        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) {
            parser_expect(parser, TK_NAME);
        }

        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            MlirValue *end_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!end_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                OperationParserResult empty = {0};
                return empty;
            }
            VecValue_push_back(parser->arena, &operands, end_operand);
        }

        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("step"))) {
            parser_expect(parser, TK_NAME);
        }

        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            MlirValue *step_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!step_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                OperationParserResult empty = {0};
                return empty;
            }
            VecValue_push_back(parser->arena, &operands, step_operand);
        }
    }

    VecValue iter_vars;
    VecValue_reserve(parser->arena, &iter_vars, 4);

    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("iter_args"))) {
        parser_expect(parser, TK_NAME);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (!parser_peek(parser, TK_REGISTER)) break;
                string iter_var_name = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);

                MlirValue *iter_var = mlir_value_create(parser->arena, BLOCK_ARG);
                mlir_value_set_register_name(iter_var, string_data_or_null(iter_var_name), iter_var_name.size);
                mlir_value_set_type(iter_var, NULL);
                VecValue_push_back(parser->arena, &iter_vars, iter_var);

                parser_expect(parser, TK_EQUAL);
                if (parser_peek(parser, TK_REGISTER)) {
                    string reg_str = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);
                    consume_optional_hash_selector(parser);
                    consume_optional_hash_selector(parser);
                    MlirValue *init_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                    if (!init_operand) {
                        parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                        OperationParserResult empty = {0};
                        return empty;
                    }
                    VecValue_push_back(parser->arena, &operands, init_operand);
                }

                if (parser_peek(parser, TK_COMMA)) {
                    parser_expect(parser, TK_COMMA);
                } else {
                    break;
                }
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    MlirType **iter_result_types = NULL;
    size_t n_iter_results = 0;
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            size_t cap = 4;
            iter_result_types = arena_alloc_array(params->arena, MlirType*, cap);
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                string t = str_lit("");
                if (!parse_type_string(parser, &t)) break;
                if (n_iter_results >= cap) {
                    cap *= 2;
                    MlirType **tmp = arena_alloc_array(params->arena, MlirType*, cap);
                    for (size_t i = 0; i < n_iter_results; i++) tmp[i] = iter_result_types[i];
                    iter_result_types = tmp;
                }
                iter_result_types[n_iter_results++] = mlir_type_create_from_string(params->arena, t);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            }
            parser_expect(parser, TK_RPAREN);
            if (n_iter_results == 0) iter_result_types = NULL;
        }
    }

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        string t = str_lit("");
        if (parse_type_string(parser, &t)) {
            iv_type = mlir_type_create_from_string(parser->arena, t);
        }
    }

    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        parser_next_token(parser);
    }
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    MlirBlock *block = mlir_block_create(parser->arena);
    if (loop_var) {
        MlirValue *iv_block_arg = mlir_value_create(parser->arena, BLOCK_ARG);
        string loop_name = mlir_value_get_register_name(loop_var);
        mlir_value_set_register_name(iv_block_arg, string_data_or_null(loop_name), loop_name.size);
        if (iv_type) {
            mlir_value_set_type(iv_block_arg, iv_type);
        } else if (operands.size > 0 && operands.data[0] && mlir_value_get_type(operands.data[0])) {
            mlir_value_set_type(iv_block_arg, mlir_value_get_type(operands.data[0]));
        } else {
            mlir_value_set_type(iv_block_arg, mlir_type_create_from_string(parser->arena, str_lit("index")));
        }
        mlir_value_set_result_index(iv_block_arg, 0);
        mlir_value_set_def(iv_block_arg, block);
        mlir_block_add_argument(parser->arena, block, iv_block_arg);
        if (loop_name.size > 0) symbol_table_add_value(parser->arena, &parser->symbol_table, loop_name, iv_block_arg);
    }

    for (size_t i = 0; i < iter_vars.size; i++) {
        MlirValue *iter_var = iter_vars.data[i];
        MlirValue *iter_block_arg = mlir_value_create(parser->arena, BLOCK_ARG);
        string iter_name = mlir_value_get_register_name(iter_var);
        mlir_value_set_register_name(iter_block_arg, string_data_or_null(iter_name), iter_name.size);
        size_t init_index = 3 + i;
        if (operands.size > init_index) {
            MlirValue *init_operand = operands.data[init_index];
            if (init_operand && mlir_value_get_type(init_operand)) {
                mlir_value_set_type(iter_block_arg, mlir_value_get_type(init_operand));
            } else {
                mlir_value_set_type(iter_block_arg, mlir_type_create_from_string(parser->arena, str_lit("unknown")));
            }
        } else {
            mlir_value_set_type(iter_block_arg, mlir_type_create_from_string(parser->arena, str_lit("unknown")));
        }
        mlir_value_set_result_index(iter_block_arg, (uint32_t)(i + 1));
        mlir_value_set_def(iter_block_arg, block);
        mlir_block_add_argument(parser->arena, block, iter_block_arg);
        if (iter_name.size > 0) symbol_table_add_value(parser->arena, &parser->symbol_table, iter_name, iter_block_arg);
    }

    bool prev_flag = parser->capture_trailing_comments;
    parser->capture_trailing_comments = true;
    while (!parser_peek(parser, TK_RBRACE)) {
        MlirOperation *inner = parse_operation(parser);
        if (parser_peek(parser, TK_NEWLINE)) {
            int64_t nl_pos = (int64_t)parser->first;
            if (nl_pos > 0) {
                int64_t line_end = nl_pos - 1;
                int64_t line_start = line_end;
                while (line_start > 0) {
                    unsigned char c = parser->input[line_start - 1];
                    if (c == '\n' || c == '\r') break;
                    line_start--;
                }
                int64_t comment_pos = -1;
                for (int64_t i = line_end - 1; i >= line_start; i--) {
                    if (parser->input[i] == '/' && parser->input[i + 1] == '/') { comment_pos = i; break; }
                }
                if (comment_pos >= 0) {
                    int64_t begin = comment_pos;
                    while (begin > line_start && parser->input[begin - 1] == ' ') begin--;
                    int64_t len = line_end - begin + 1;
                    if (len > 0) {
                        string comment = str_from_cstr_len_view((char*)parser->input + begin, len);
                        mlir_operation_set_trailing_comment(inner, comment.str, comment.size);
                    }
                }
            }
        } else {
            string text = str_from_cstr_view((char*)parser->input);
            int64_t start = (int64_t)parser->last + 1;
            if (start < (int64_t)text.size) {
                int64_t end = (int64_t)text.size - 1;
                for (int64_t i = start; i < (int64_t)text.size; i++) {
                    char ch = text.str[i];
                    if (ch == '\n' || ch == '\r') { end = i - 1; break; }
                }
                if (end >= start) {
                    int64_t cpos = -1;
                    for (int64_t i = end - 1; i >= start; i--) {
                        if (text.str[i] == '/' && text.str[i+1] == '/') { cpos = i; break; }
                    }
                    if (cpos >= 0) {
                        int64_t begin = cpos;
                        while (begin > start && text.str[begin - 1] == ' ') begin--;
                        int64_t len = end - begin + 1;
                        if (len > 0) {
                            string c = str_from_cstr_len_view(text.str + begin, len);
                            mlir_operation_set_trailing_comment(inner, c.str, c.size);
                        }
                    }
                }
            }
        }
        mlir_block_add_operation(parser->arena, block, inner);
        parser_expect(parser, TK_NEWLINE);
        while (parser_peek(parser, TK_NEWLINE)) parser_expect(parser, TK_NEWLINE);
    }
    parser_expect(parser, TK_RBRACE);
    parser->capture_trailing_comments = prev_flag;

    symbol_table_pop_scope(&parser->symbol_table);

    MlirRegion *region = mlir_region_create(parser->arena);
    mlir_region_add_block(parser->arena, region, block);

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    MlirRegion **regions = arena_alloc_array(params->arena, MlirRegion*, 1);
    regions[0] = region;

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        NULL,
        0,
        iter_result_types,
        n_iter_results,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        regions,
        1,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    MlirValue **results = NULL;
    size_t n_results = 0;
    if (n_iter_results > 0 && params->n_lhs_results > 0) {
        // lhs_results now contains the correct count matching n_iter_results
        // (expanded when parsing %reg:N syntax)
        results = params->lhs_results;
        n_results = params->n_lhs_results;

        // Set def and types on named results; set unnamed results to NULL so they print as %_
        for (size_t i = 0; i < n_results; i++) {
            string reg_name = mlir_value_get_register_name(results[i]);
            if (reg_name.size > 0) {
                // Named result: set def and type
                mlir_value_set_def(results[i], op);
                mlir_value_set_type(results[i], iter_result_types[i]);
            } else {
                // Unnamed result: set to NULL so it prints as %_
                results[i] = NULL;
            }
        }

        // Use consolidated API now that counts match
        mlir_operation_set_results_with_types(op, results, iter_result_types, n_results);
    }

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}
OperationParserResult parse_scf_while_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        bool first = true;
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (!first && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            first = false;
            if (!parser_peek(parser, TK_REGISTER)) break;
            parser_expect(parser, TK_REGISTER);
            parser_expect(parser, TK_EQUAL);
            if (parser_peek(parser, TK_REGISTER)) {
                string reg = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *init = symbol_table_lookup(&parser->symbol_table, reg);
                if (!init) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    OperationParserResult empty = {0};
                    return empty;
                }
                VecValue_push_back(parser->arena, &operands, init);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            int depth = 1;
            while (depth > 0 && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LPAREN)) depth++;
                else if (parser_peek(parser, TK_RPAREN)) depth--;
                parser_next_token(parser);
            }
            if (parser_peek(parser, TK_RPAREN)) parser_expect(parser, TK_RPAREN);
        } else {
            string tmp = str_lit("");
            (void)parse_type_string(parser, &tmp);
        }
    }

    MlirType **result_types = NULL;
    size_t n_result_types = 0;
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            size_t cap = 2;
            result_types = arena_alloc_array(params->arena, MlirType*, cap);
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                string t = str_lit("");
                if (!parse_type_string(parser, &t)) break;
                if (n_result_types >= cap) {
                    cap *= 2;
                    MlirType **tmp = arena_alloc_array(params->arena, MlirType*, cap);
                    for (size_t i = 0; i < n_result_types; i++) tmp[i] = result_types[i];
                    result_types = tmp;
                }
                result_types[n_result_types++] = mlir_type_create_from_string(params->arena, t);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            }
            parser_expect(parser, TK_RPAREN);
            if (n_result_types == 0) result_types = NULL;
        } else {
            string t = str_lit("");
            if (parse_type_string(parser, &t)) {
                result_types = arena_alloc_array(params->arena, MlirType*, 1);
                result_types[0] = mlir_type_create_from_string(params->arena, t);
                n_result_types = 1;
            }
        }
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        parser_next_token(parser);
    }
    MlirRegion *cond_region = parse_region(parser);

    MlirRegion *body_region = NULL;
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("do"))) {
        parser_expect(parser, TK_NAME);
        body_region = parse_region(parser);
    }

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    size_t n_regions = body_region ? 2 : 1;
    MlirRegion **regions = arena_alloc_array(params->arena, MlirRegion*, n_regions);
    regions[0] = cond_region;
    if (body_region) regions[1] = body_region;

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        regions,
        n_regions,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

void parse_affine_for(Parser *parser, MlirOperation *op) {
    // Expect induction variable
    MlirValue *ind_var = NULL;
    if (parser_peek(parser, TK_REGISTER)) {
        string iv_name = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);

        // Create a placeholder for the block argument; registered later
        ind_var = mlir_value_create(parser->arena, BLOCK_ARG);
        mlir_value_set_register_name(ind_var, string_data_or_null(iv_name), iv_name.size);
        mlir_value_set_type(ind_var, mlir_type_create_from_string(parser->arena, str_lit("index")));


        // '=' lower bound
        if (parser_peek(parser, TK_EQUAL)) {
            parser_expect(parser, TK_EQUAL);
            // Consume a simple affine lower bound: integer or register/expression until 'to'
            while (!parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) break;
                if (parser_peek(parser, TK_LBRACE_END)) break;
                parser_next_token(parser);
            }
        }

        // 'to' upper bound
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) {
            parser_expect(parser, TK_NAME);
            // Consume simple upper bound until optional 'step' or '{'
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("step"))) break;
                parser_next_token(parser);
            }
        }

        // Optional 'step' expression
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("step"))) {
            parser_expect(parser, TK_NAME);
            // Consume step expression until '{'
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
                parser_next_token(parser);
            }
        }
    }

    // Now parse the loop region
    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        parser_next_token(parser);
    }
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    // Push new scope for induction variable
    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    // Create single block and register induction variable as block argument
    MlirBlock *block = mlir_block_create(parser->arena);
    if (ind_var) {
        MlirValue *iv_block_arg = mlir_value_create(parser->arena, BLOCK_ARG);
        string block_name = str_lit("%arg0");
        mlir_value_set_register_name(iv_block_arg, block_name.str, block_name.size);
        mlir_value_set_type(iv_block_arg, mlir_type_create_from_string(parser->arena, str_lit("index")));
        mlir_value_set_result_index(iv_block_arg, (uint32_t)mlir_block_num_arguments(block));
        mlir_value_set_def(iv_block_arg, block);
        mlir_block_add_argument(parser->arena, block, iv_block_arg);
        string orig_name = mlir_value_get_register_name(ind_var);
        if (orig_name.size > 0) symbol_table_add_value(parser->arena, &parser->symbol_table, orig_name, iv_block_arg);
    }

    // Parse body operations
    bool prev_flag = parser->capture_trailing_comments;
    parser->capture_trailing_comments = true;
    while (!parser_peek(parser, TK_RBRACE)) {
        MlirOperation *inner = parse_operation(parser);
        // Capture trailing inline comment before consuming the newline
        if (parser_peek(parser, TK_NEWLINE)) {
            int64_t nl_pos = (int64_t)parser->first;
            if (nl_pos > 0) {
                int64_t line_end = nl_pos - 1;
                int64_t line_start = line_end;
                while (line_start > 0) {
                    unsigned char c = parser->input[line_start - 1];
                    if (c == '\n' || c == '\r') break;
                    line_start--;
                }
                int64_t comment_pos = -1;
                for (int64_t i = line_end - 1; i >= line_start; i--) {
                    if (parser->input[i] == '/' && parser->input[i + 1] == '/') { comment_pos = i; break; }
                }
                if (comment_pos >= 0) {
                    int64_t begin = comment_pos;
                    while (begin > line_start && parser->input[begin - 1] == ' ') begin--;
                    int64_t len = line_end - begin + 1;
                    if (len > 0) {
                        string c = str_from_cstr_len_view((char*)parser->input + begin, len);
                        mlir_operation_set_trailing_comment(inner, c.str, c.size);
                    }
                }
            }
        } else {
            string text = str_from_cstr_view((char*)parser->input);
            int64_t start = (int64_t)parser->last + 1;
            if (start < (int64_t)text.size) {
                int64_t end = (int64_t)text.size - 1;
                for (int64_t i = start; i < (int64_t)text.size; i++) {
                    char ch = text.str[i];
                    if (ch == '\n' || ch == '\r') { end = i - 1; break; }
                }
                if (end >= start) {
                    int64_t cpos = -1;
                    for (int64_t i = end - 1; i >= start; i--) {
                        if (text.str[i] == '/' && text.str[i+1] == '/') { cpos = i; break; }
                    }
                    if (cpos >= 0) {
                        int64_t begin = cpos;
                        while (begin > start && text.str[begin - 1] == ' ') begin--;
                        int64_t len = end - begin + 1;
                        if (len > 0) {
                            string c = str_from_cstr_len_view(text.str + begin, len);
                            mlir_operation_set_trailing_comment(inner, c.str, c.size);
                        }
                    }
                }
            }
        }
        mlir_block_add_operation(parser->arena, block, inner);
        parser_expect(parser, TK_NEWLINE);
        while (parser_peek(parser, TK_NEWLINE)) parser_expect(parser, TK_NEWLINE);
    }
    parser_expect(parser, TK_RBRACE);
    parser->capture_trailing_comments = prev_flag;

    // Optional trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        mlir_operation_set_location(op, parse_loc(parser));
    }

    symbol_table_pop_scope(&parser->symbol_table);
    MlirRegion *region = mlir_region_create(parser->arena);
    mlir_region_add_block(parser->arena, region, block);
    mlir_op_add_region(parser->arena, op, region);
}

OperationParserResult parse_gpu_launch_op(Parser *parser, const OperationParserParams *params) {
    // Parse gpu.launch blocks(%arg2, %arg3, %arg4) in (...) threads(%arg5, %arg6, %arg7) in (...) {

    VecValue launch_args;
    VecValue_reserve(parser->arena, &launch_args, 16);

    // Parse blocks(...) section
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("blocks"))) {
        parser_expect(parser, TK_NAME);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);

            // Parse block arguments
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string reg_str = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);

                    MlirValue *arg = mlir_value_create(parser->arena, BLOCK_ARG);
                    mlir_value_set_register_name(arg, string_data_or_null(reg_str), reg_str.size);
                    mlir_value_set_result_index(arg, (uint32_t)launch_args.size);
                    mlir_value_set_type(arg, mlir_type_create_from_string(parser->arena, str_lit("index")));

                    VecValue_push_back(parser->arena, &launch_args, arg);

                    if (parser_peek(parser, TK_COMMA)) {
                        parser_expect(parser, TK_COMMA);
                    }
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    // Skip until threads(...) section
    while (!parser_peek(parser, TK_EOF) && !(parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("threads")))) {
        parser_next_token(parser);
    }

    // Parse threads(...) section
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("threads"))) {
        parser_expect(parser, TK_NAME);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);

            // Parse thread arguments
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string reg_str = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);

                    MlirValue *arg = mlir_value_create(parser->arena, BLOCK_ARG);
                    mlir_value_set_register_name(arg, string_data_or_null(reg_str), reg_str.size);
                    mlir_value_set_result_index(arg, (uint32_t)launch_args.size);
                    mlir_value_set_type(arg, mlir_type_create_from_string(parser->arena, str_lit("index")));

                    VecValue_push_back(parser->arena, &launch_args, arg);

                    if (parser_peek(parser, TK_COMMA)) {
                        parser_expect(parser, TK_COMMA);
                    }
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    // Parse any attributes and result types using helper functions
    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirType **result_types = NULL;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, NULL);

    // Parse optional location
    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // Skip until opening brace
    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        parser_next_token(parser);
    }

    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) parser_next_token(parser);

    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    MlirBlock *gpu_block = mlir_block_create(parser->arena);
    for (size_t i = 0; i < launch_args.size; i++) {
        MlirValue *arg = launch_args.data[i];
        mlir_value_set_def(arg, gpu_block);
        mlir_block_add_argument(parser->arena, gpu_block, arg);
    }

    symbol_table_push_scope(parser->arena, &parser->symbol_table);
    for (size_t i = 0; i < launch_args.size; i++) {
        string name = mlir_value_get_register_name(launch_args.data[i]);
        if (name.size > 0) symbol_table_add_value(parser->arena, &parser->symbol_table, name, launch_args.data[i]);
    }

    while (!parser_peek(parser, TK_RBRACE)) {
        MlirOperation *inner_op = parse_operation(parser);
        mlir_block_add_operation(parser->arena, gpu_block, inner_op);
        parser_expect(parser, TK_NEWLINE);
        while (parser_peek(parser, TK_NEWLINE)) parser_expect(parser, TK_NEWLINE);
    }
    parser_expect(parser, TK_RBRACE);
    symbol_table_pop_scope(&parser->symbol_table);

    MlirRegion *gpu_region = mlir_region_create(parser->arena);
    mlir_region_add_block(parser->arena, gpu_region, gpu_block);

    // Create the operation at the end
    MlirOperation *op = mlir_op_create(params->arena, params->op_type, str_lit(""),
                                      attributes, n_attributes,
                                      result_types, n_result_types,
                                      params->lhs_results, params->n_lhs_results,
                                      NULL, 0,
                                      NULL, 0,
                                      op_location, params->unnumbered_loc_def,
                                      str_lit(""), params->source_line_start);

    mlir_op_add_region(parser->arena, op, gpu_region);

    // Create results from the operation's result types
    MlirValue **results = NULL;
    size_t n_results = 0;

    results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_arith_cmpi_op(Parser *parser, const OperationParserParams *params) {
    // Parse arith.cmpi <predicate>, %lhs, %rhs : type
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);

    // Parse predicate (slt, sge, etc.)
    string predicate = str_lit("slt"); // default
    if (parser_peek(parser, TK_NAME)) {
        predicate = parser_token_str(parser);
        parser_expect(parser, TK_NAME);
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0, attributes_capacity = 0;
    append_attr(parser, &attributes, &n_attributes, &attributes_capacity, create_string_attr(parser, str_lit("predicate"), predicate));

    // Expect comma
    parser_expect(parser, TK_COMMA);

    // Parse operands
    if (parser_peek(parser, TK_REGISTER)) {
        string lhs_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *lhs = symbol_table_lookup(&parser->symbol_table, lhs_str);
        if (!lhs) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, lhs);
    }

    parser_expect(parser, TK_COMMA);

    if (parser_peek(parser, TK_REGISTER)) {
        string rhs_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *rhs = symbol_table_lookup(&parser->symbol_table, rhs_str);
        if (!rhs) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, rhs);
    }

    // Parse any additional attributes and result types using helper functions
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirType **result_types = NULL;
    size_t n_result_types = 0;
    parse_result_types(parser, &result_types, &n_result_types, &attributes, &n_attributes, &attributes_capacity, params->op_type, NULL);

    // Default result type if not specified
    if (n_result_types == 0) {
        result_types = arena_alloc_array(params->arena, MlirType*, 1);
        result_types[0] = mlir_type_create_from_string(params->arena, str_lit("i1"));
        n_result_types = 1;
    }

    // Parse optional location
    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) {
        op_location = params->unnumbered_loc_def;
    }

    // Create the operation at the end
    MlirOperation *op = mlir_op_create(params->arena, params->op_type, str_lit(""),
                                      attributes, n_attributes,
                                      result_types, n_result_types,
                                      params->lhs_results, params->n_lhs_results,
                                      operands.data, operands.size,
                                      NULL, 0,
                                      op_location, params->unnumbered_loc_def,
                                      str_lit(""), params->source_line_start);

    // Create results from the operation's result types
    MlirValue **results = NULL;
    size_t n_results = 0;

    results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}


OperationParserResult parse_scf_yield_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);

    while (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, operand);
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        else break;
    }

    // Handle optional type information after all operands
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        // Parse type list until we hit loc() or end
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                break;
            }
            parser_next_token(parser);
        }
    }

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) op_location = params->unnumbered_loc_def;

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        NULL, 0,
        NULL, 0,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        NULL, 0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = NULL,
        .n_results = 0,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_return_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);

    // Handle optional parentheses
    bool has_parens = false;
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        has_parens = true;
    }

    while (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            OperationParserResult empty = {0};
            return empty;
        }
        VecValue_push_back(parser->arena, &operands, operand);
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        else break;
    }

    if (has_parens) {
        parser_expect(parser, TK_RPAREN);
    }

    // Handle optional type information
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        // Skip type information
        while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_RBRACE)) {
            parser_next_token(parser);
        }
    }

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) op_location = params->unnumbered_loc_def;

    bool keep_operands = params->op_type != OP_TYPE_STD_RETURN;

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        NULL, 0,
        NULL, 0,
        params->lhs_results,
        params->n_lhs_results,
        (keep_operands && operands.size > 0) ? operands.data : NULL,
        keep_operands ? operands.size : 0,
        NULL, 0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = NULL,
        .n_results = 0,
        .location = op_location
    };
    return out;
}


OperationParserResult parse_tt_load_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 3);

    while (parser_peek(parser, TK_REGISTER)) {
        if (!parse_register_operand(parser, &operands, true)) {
            OperationParserResult empty = {0};
            return empty;
        }
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        else break;
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirType **result_types = NULL;
    size_t n_result_types = 0;

    while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);

    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);

        string pointer_sig = str_lit("");
        if (parse_type_string(parser, &pointer_sig)) {
            string result_sig = pointer_sig;

            while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);
            if (parser_peek(parser, TK_ARROW)) {
                parser_expect(parser, TK_ARROW);
                while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);
                string explicit_result = str_lit("");
                if (parse_type_string(parser, &explicit_result)) {
                    result_sig = explicit_result;
                }
            } else {
                size_t pos = (size_t)-1;
                const size_t needle_len = 8; // strlen("!tt.ptr<")
                for (size_t i = 0; i + needle_len <= pointer_sig.size; i++) {
                    if (pointer_sig.str[i] == '!' &&
                        pointer_sig.str[i + 1] == 't' &&
                        pointer_sig.str[i + 2] == 't' &&
                        pointer_sig.str[i + 3] == '.' &&
                        pointer_sig.str[i + 4] == 'p' &&
                        pointer_sig.str[i + 5] == 't' &&
                        pointer_sig.str[i + 6] == 'r' &&
                        pointer_sig.str[i + 7] == '<') {
                        pos = i;
                        break;
                    }
                }

                if (pos != (size_t)-1) {
                    size_t start = pos + needle_len;
                    size_t end = start;
                    while (end < pointer_sig.size && pointer_sig.str[end] != '>' && pointer_sig.str[end] != ',') end++;
                    string elem = str_substr(pointer_sig, start, end - start);

                    size_t close = end;
                    while (close < pointer_sig.size && pointer_sig.str[close] != '>') close++;
                    if (close < pointer_sig.size) close++;

                    string before = str_substr(pointer_sig, 0, pos);
                    string after = str_substr(pointer_sig, close, pointer_sig.size - close);

                    string val_sig = before;
                    val_sig = str_concat(parser->arena, val_sig, elem.size > 0 ? elem : str_lit("f32"));
                    val_sig = str_concat(parser->arena, val_sig, after);
                    result_sig = val_sig;
                }
            }

            if (result_sig.size > 0) {
                result_types = arena_alloc_array(params->arena, MlirType*, 1);
                result_types[0] = mlir_type_create_from_string(params->arena, result_sig);
                n_result_types = 1;
            }
        }

        while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);
        if (parser_peek(parser, TK_COMMA)) {
            do {
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            } while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE));
        }
    }

    while (parser_peek(parser, TK_WHITESPACE)) parser_expect(parser, TK_WHITESPACE);
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);
    parse_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) op_location = params->unnumbered_loc_def;

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        result_types,
        n_result_types,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        NULL, 0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    size_t n_results = 0;
    MlirValue **results = finalize_results(params, op, result_types, n_result_types, &n_results);

    OperationParserResult out = {
        .operation = op,
        .results = results,
        .n_results = n_results,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_tt_store_op(Parser *parser, const OperationParserParams *params) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 3);

    // Parse operands: pointer, value, mask
    while (parser_peek(parser, TK_REGISTER)) {
        if (!parse_register_operand(parser, &operands, true)) {
            OperationParserResult empty = {0};
            return empty;
        }
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        else break;
    }

    MlirAttribute **attributes = NULL;
    size_t n_attributes = 0;
    size_t attributes_capacity = 0;

    // Custom attribute parsing for tt.store to handle ellipsis and create proper integer attributes
    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        while (!parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_NAME)) {
                string attr_name = parser_token_str(parser);
                parser_expect(parser, TK_NAME);
                parser_expect(parser, TK_EQUAL);

                if (parser_peek(parser, TK_DOT)) {
                    // Handle "..." (ellipsis) tokens - convert to integer 1
                    parser_expect(parser, TK_DOT);
                    parser_expect(parser, TK_DOT);
                    parser_expect(parser, TK_DOT);

                    append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                                create_integer_attr(parser, attr_name, 1));
                } else if (parser_peek(parser, TK_INTEGER)) {
                    // Handle regular integer values
                    string ival = parser_token_str(parser);
                    char *buffer = arena_alloc_array(parser->arena, char, ival.size + 1);
                    memcpy(buffer, ival.str, ival.size);
                    buffer[ival.size] = '\0';
                    int64_t int_val = strtoll(buffer, NULL, 10);
                    parser_expect(parser, TK_INTEGER);

                    // Skip optional type annotation (:i32)
                    if (parser_peek(parser, TK_COLON)) {
                        parser_expect(parser, TK_COLON);
                        parser_expect(parser, TK_NAME);
                    }

                    append_attr(parser, &attributes, &n_attributes, &attributes_capacity,
                                create_integer_attr(parser, attr_name, int_val));
                } else {
                    // Skip unknown tokens
                    parser_next_token(parser);
                }
            }
            if (parser_peek(parser, TK_COMMA)) {
                parser_expect(parser, TK_COMMA);
            }
        }
        parser_expect(parser, TK_RBRACE);
    }
    parse_angle_brace_attributes(parser, &attributes, &n_attributes, &attributes_capacity);

    // Handle optional type information
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        // Skip type information for store operations
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                break;
            }
            parser_next_token(parser);
        }
    }

    MlirLocation *op_location = parse_optional_location(parser);
    if (!op_location) op_location = params->unnumbered_loc_def;

    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        attributes,
        n_attributes,
        NULL, 0,
        params->lhs_results,
        params->n_lhs_results,
        operands.data,
        operands.size,
        NULL, 0,
        op_location,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);

    OperationParserResult out = {
        .operation = op,
        .results = NULL,
        .n_results = 0,
        .location = op_location
    };
    return out;
}

OperationParserResult parse_func_func_op(Parser *parser, const OperationParserParams *params) {
    // For now, create a minimal implementation that delegates to the old parser
    // This is a complex function that would need significant work to properly migrate
    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        NULL, 0,
        NULL, 0,
        params->lhs_results,
        params->n_lhs_results,
        NULL, 0,
        NULL, 0,
        params->unnumbered_loc_def,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);
    parse_func_func(parser, op);
    parse_generic_attrs_and_result_type(parser, op);

    OperationParserResult out = {
        .operation = op,
        .results = NULL,
        .n_results = 0,
        .location = params->unnumbered_loc_def
    };
    return out;
}

OperationParserResult parse_affine_for_op(Parser *parser, const OperationParserParams *params) {
    // For now, create a minimal implementation that delegates to the old parser
    // This is a complex function that would need significant work to properly migrate
    MlirOperation *op = mlir_op_create(
        params->arena,
        params->op_type,
        str_lit(""),
        NULL, 0,
        NULL, 0,
        params->lhs_results,
        params->n_lhs_results,
        NULL, 0,
        NULL, 0,
        params->unnumbered_loc_def,
        params->unnumbered_loc_def,
        str_lit(""),
        params->source_line_start);
    parse_affine_for(parser, op);
    parse_generic_attrs_and_result_type(parser, op);

    OperationParserResult out = {
        .operation = op,
        .results = NULL,
        .n_results = 0,
        .location = params->unnumbered_loc_def
    };
    return out;
}

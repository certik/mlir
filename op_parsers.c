#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <base/string.h>
#include <base/io.h>
#include <base/vector.h>

#include "mlir_parser.h"
#include "mlir_ir_internal.h"
#include "tokenizer.h"
#include "op_parsers.h"

// Helper function declarations
static bool parse_type_string(Parser *parser, string *out) {
    if (!(parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION))) {
        return false;
    }

    string ty = parser_token_str(parser);
    parser_next_token(parser);

    // Dialect types starting with '!'
    if (ty.size == 1 && ty.str[0] == '!' && (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME))) {
        ty = str_concat(parser->arena, ty, parser_token_str(parser));
        parser_next_token(parser);
    }

    // Optional angle-bracketed content (supports nesting)
    if (parser_peek(parser, TK_LANGLE)) {
        int depth = 0;
        do {
            if (parser_peek(parser, TK_LANGLE)) depth++;
            else if (parser_peek(parser, TK_RANGLE)) depth--;
            ty = str_concat(parser->arena, ty, parser_token_str(parser));
            parser_next_token(parser);
        } while (depth > 0 && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE));
    }

    *out = ty;
    return true;
}

// Helper function to consume optional hash selector
void consume_optional_hash_selector(Parser *parser) {
    // Consume tokens like #0, #1 after a register (e.g., %49#0) and ignore them
    if (parser_peek(parser, TK_HASH_NAME)) {
        parser_next_token(parser);
    }
}

static void set_op_attributes(MlirOperation *op, MlirAttribute **attrs, size_t count) {
    mlir_operation_set_attributes(op, attrs, count);
}

static void set_op_operands(MlirOperation *op, MlirValue **operands, size_t count) {
    mlir_operation_set_operands(op, operands, count);
}

static void set_op_result_types(MlirOperation *op, MlirType **types, size_t count) {
    mlir_operation_set_result_types(op, types, count);
}

static inline const char *string_data_or_null(string s) {
    return s.size > 0 ? s.str : NULL;
}

static MlirAttribute *create_string_attr(Parser *parser, string name, string value) {
    MlirAttribute *attr = mlir_attribute_create_string(parser->arena, string_data_or_null(value), value.size);
    if (name.size > 0) mlir_attribute_set_name(attr, string_data_or_null(name), name.size);
    return attr;
}

static MlirAttribute *create_integer_attr(Parser *parser, string name, int64_t value) {
    MlirAttribute *attr = mlir_attribute_create_integer(parser->arena, value);
    if (name.size > 0) mlir_attribute_set_name(attr, string_data_or_null(name), name.size);
    return attr;
}

static MlirAttribute *create_float_attr(Parser *parser, string name, double value) {
    MlirAttribute *attr = mlir_attribute_create_float(parser->arena, value);
    if (name.size > 0) mlir_attribute_set_name(attr, string_data_or_null(name), name.size);
    return attr;
}

static MlirAttribute *create_bool_attr(Parser *parser, string name, bool value) {
    MlirAttribute *attr = mlir_attribute_create_bool(parser->arena, value);
    if (name.size > 0) mlir_attribute_set_name(attr, string_data_or_null(name), name.size);
    return attr;
}

static void operation_append_attribute(Parser *parser, MlirOperation *op, MlirAttribute *attr) {
    if (!attr) return;
    mlir_operation_append_attribute(parser->arena, op, attr);
}

static void operation_set_single_attribute(Parser *parser, MlirOperation *op, MlirAttribute *attr) {
    if (!attr) return;
    MlirAttribute **attrs = arena_alloc_array(parser->arena, MlirAttribute*, 1);
    attrs[0] = attr;
    set_op_attributes(op, attrs, 1);
}

static MlirValue *lookup_or_create_value(Parser *parser, string reg, string default_type) {
    MlirValue *val = symbol_table_lookup(&parser->symbol_table, reg);
    if (!val) {
        val = create_value_ref(parser->arena, BLOCK_ARG);
        mlir_value_set_register_name(val, string_data_or_null(reg), reg.size);
        if (default_type.size > 0) {
            MlirType *ty = mlir_type_create_from_string(parser->arena, default_type);
            mlir_value_set_type(val, ty);
        }
    }
    return val;
}

static void append_attr(Parser *parser, MlirAttribute ***attrs, size_t *n, size_t *cap, MlirAttribute *attr) {
    if (!attr) return;
    size_t new_cap = (*cap == 0) ? 4 : *cap;
    if (*attrs == NULL) {
        *attrs = arena_alloc_array(parser->arena, MlirAttribute*, new_cap);
        *cap = new_cap;
    } else if (*n >= *cap) {
        new_cap = (*cap) * 2;
        MlirAttribute **new_attrs = arena_alloc_array(parser->arena, MlirAttribute*, new_cap);
        for (size_t i = 0; i < *n; i++) new_attrs[i] = (*attrs)[i];
        *attrs = new_attrs;
        *cap = new_cap;
    }
    (*attrs)[(*n)++] = attr;
}

static void attr_list_init_from_op(Parser *parser, MlirOperation *op, MlirAttribute ***attrs, size_t *n, size_t *cap) {
    if (op->attributes && op->n_attributes > 0) {
        *cap = op->n_attributes + 4;
        *attrs = arena_alloc_array(parser->arena, MlirAttribute*, *cap);
        for (size_t i = 0; i < op->n_attributes; i++) (*attrs)[i] = op->attributes[i];
        *n = op->n_attributes;
    }
}

// Include the extracted functions
void parse_generic_attrs_and_result_type(Parser *parser, MlirOperation *op) {
    // Attributes block
    // Handle generic attributes in angle-bracket form: <{ ... }>
    if (parser_peek(parser, TK_LANGLE)) {
        // Lookahead for '<{' sequence
        uint64_t save_first = parser->first, save_last = parser->last, save_cur = parser->cur; TokenType save_sym = parser->sym;
        parser_expect(parser, TK_LANGLE);
        bool had_attrs=false;
        if (parser_peek(parser, TK_LBRACE)) {
            parser_expect(parser, TK_LBRACE);
            MlirAttribute **attrs = NULL; size_t n_attrs=0; size_t cap=4; attrs = arena_alloc_array(parser->arena, MlirAttribute*, cap);
            while (!parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                    string attr_name = parser_token_str(parser); parser_next_token(parser);
                    if (parser_peek(parser, TK_EQUAL)) { parser_expect(parser, TK_EQUAL);
                        if (n_attrs>=cap){ cap*=2; MlirAttribute**na = arena_alloc_array(parser->arena, MlirAttribute*, cap); for(size_t i=0;i<n_attrs;i++) na[i]=attrs[i]; attrs=na; }
                        string payload = str_lit(""); int angle=0; while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_COMMA) && !parser_peek(parser, TK_RBRACE)) { string tok=parser_token_str(parser); if (parser_peek(parser, TK_LANGLE)) angle++; else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--; payload = payload.size ? str_concat(parser->arena, payload, tok) : tok; parser_next_token(parser); if (angle==0 && (parser_peek(parser, TK_COMMA) || parser_peek(parser, TK_RBRACE))) break; }
                        attrs[n_attrs++] = create_string_attr(parser, attr_name, payload); had_attrs=true;
                    }
                    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
                } else { parser_next_token(parser); }
            }
            if (parser_peek(parser, TK_RBRACE)) parser_expect(parser, TK_RBRACE);
            if (had_attrs) { set_op_attributes(op, attrs, n_attrs); }
            if (parser_peek(parser, TK_RANGLE)) parser_expect(parser, TK_RANGLE);
        } else {
            // Not actually an attribute block; rewind
            parser->first = save_first; parser->last = save_last; parser->cur = save_cur; parser->sym = save_sym;
        }
    }
    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        
        MlirAttribute **attrs = NULL;
        size_t n_attrs = 0;
        size_t attrs_capacity = 4;
        attrs = arena_alloc_array(parser->arena, MlirAttribute*, attrs_capacity);
        
        while (!parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF)) {
            // Parse attribute name
            if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                string attr_name = parser_token_str(parser);
                parser_next_token(parser);
                
                // Expect '='
                if (parser_peek(parser, TK_EQUAL)) {
                    parser_expect(parser, TK_EQUAL);
                    
                    // Grow array if needed
                    if (n_attrs >= attrs_capacity) {
                        attrs_capacity *= 2;
                        MlirAttribute **new_attrs = arena_alloc_array(parser->arena, MlirAttribute*, attrs_capacity);
                        for (size_t i = 0; i < n_attrs; i++) new_attrs[i] = attrs[i];
                        attrs = new_attrs;
                    }
                    
                    MlirAttribute *attr = arena_alloc(parser->arena, struct MlirAttribute);
                    attr->name = attr_name;
                    
                    // Parse attribute value: capture complex payload verbatim until ',' or '}'
                    string payload = str_lit("");
                    int angle = 0;
                    while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_COMMA) && !parser_peek(parser, TK_RBRACE)) {
                        string tok = parser_token_str(parser);
                        if (parser_peek(parser, TK_LANGLE)) angle++;
                        else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--;
                        payload = payload.size ? str_concat(parser->arena, payload, tok) : tok;
                        parser_next_token(parser);
                        if (angle==0 && (parser_peek(parser, TK_COMMA) || parser_peek(parser, TK_RBRACE))) break;
                    }
                    attr->kind = ATTR_KIND_STRING;
                    attr->data.string_value = payload;
                    
                    attrs[n_attrs++] = attr;
                }
                
                // Skip comma if present
                if (parser_peek(parser, TK_COMMA)) {
                    parser_expect(parser, TK_COMMA);
                }
            } else {
                // Skip unknown token
                parser_next_token(parser);
            }
        }
        
        if (parser_peek(parser, TK_RBRACE)) {
            parser_expect(parser, TK_RBRACE);
        }
        
        // Store parsed attributes
        if (n_attrs > 0) {
            set_op_attributes(op, attrs, n_attrs);
        }
    }

    // Optional result type after ':'
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        // Handle forms like ": (type, ...) -> type"
        if (parser_peek(parser, TK_LPAREN)) {
            // Parse and capture the first type inside parentheses for classic printing
            parser_expect(parser, TK_LPAREN);
            string src_sig = str_lit("");
            (void)parse_type_string(parser, &src_sig);
            // Consume until matching ')'
            int depth = 1;
            while (depth > 0 && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LPAREN)) depth++;
                else if (parser_peek(parser, TK_RPAREN)) depth--;
                parser_next_token(parser);
            }

            // Optional arrow and a result type
            if (parser_peek(parser, TK_ARROW)) {
                parser_expect(parser, TK_ARROW);
            }

            string type_str = str_lit("");
            if (parse_type_string(parser, &type_str)) {
                MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
                types[0] = mlir_type_create_from_string(parser->arena, type_str);
                set_op_result_types(op, types, 1);
            }

            // Record that the signature used parenthesized operand types
            operation_append_attribute(parser, op, create_bool_attr(parser, str_lit("_sig_parens"), true));
            operation_append_attribute(parser, op, create_string_attr(parser, str_lit("_sig_src"), src_sig));

            // Optional trailing loc()
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                MlirLocation *loc = parse_loc(parser);
                mlir_operation_set_location(op, loc);
            }
            return;
        }

        if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME) || parser_peek(parser, TK_EXCLAMATION)) {
            // Parse a type token (could be src type or result type)
            string type_left = str_lit("");
            parse_type_string(parser, &type_left);

            // Handle special form: "type * type -> result"
            if (parser_peek(parser, TK_STAR)) {
                parser_expect(parser, TK_STAR);
                string type_right_mul = str_lit("");
                parse_type_string(parser, &type_right_mul);
                // Expect arrow and a result type
                if (parser_peek(parser, TK_ARROW)) parser_expect(parser, TK_ARROW);
                string type_res = str_lit("");
                if (parse_type_string(parser, &type_res)) {
                    MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
                    types[0] = mlir_type_create_from_string(parser->arena, type_res);
                    set_op_result_types(op, types, 1);
                }
                // Done handling this signature. Do not consume further (keep trailing loc()).
                return;
            } else if (parser_peek(parser, TK_ARROW)) {
                // Form ": <src-type> -> <result-type>"
                parser_expect(parser, TK_ARROW);
                string type_right = str_lit("");
                if (parse_type_string(parser, &type_right)) {
                    MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
                    types[0] = mlir_type_create_from_string(parser->arena, type_right);
                    set_op_result_types(op, types, 1);
                }
                // Record source signature string for classic printing
                operation_append_attribute(parser, op, create_string_attr(parser, str_lit("_sig_src"), type_left));
            } else if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) {
                // Casts like ": src_ty to dst_ty"; set result to dst_ty
                parser_expect(parser, TK_NAME);
                string type_dst = str_lit("");
                if (parse_type_string(parser, &type_dst)) {
                    MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
                    types[0] = mlir_type_create_from_string(parser->arena, type_dst);
                    set_op_result_types(op, types, 1);
                }
            } else if (parser_peek(parser, TK_COMMA)) {
                // Operand type list ": type, type, ..." — consume conservatively
                while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
                    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
                    parser_next_token(parser);
                }
            } else {
                // Treat as single result type ": type" for most ops,
                // but do NOT override for compare ops like arith.cmpi where
                // the colon type is the operand type, not the result.
                if (mlir_operation_get_type(op) != OP_TYPE_ARITH_CMPI) {
                    MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
                    types[0] = mlir_type_create_from_string(parser->arena, type_left);
                    set_op_result_types(op, types, 1);
                }
            }
        }
    }

    // Or optional result type directly after operands via '-> type'
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        string type_str = str_lit("");
        if (parse_type_string(parser, &type_str)) {
            MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
            types[0] = mlir_type_create_from_string(parser->arena, type_str);
            set_op_result_types(op, types, 1);
        }
    }

    // Optional trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        MlirLocation *loc = parse_loc(parser);
        mlir_operation_set_location(op, loc);
    }
}

void parse_arith_constant(Parser *parser, MlirOperation *op) {
    // arith.constant value : type
    if (parser_peek(parser, TK_INTEGER)) {
        string value_str = parser_token_str(parser);
        parser_expect(parser, TK_INTEGER);

        int64_t parsed_value = 0;
        for (size_t i = 0; i < value_str.size; i++) {
            if (value_str.str[i] >= '0' && value_str.str[i] <= '9') {
                parsed_value = parsed_value * 10 + (value_str.str[i] - '0');
            }
        }
        operation_set_single_attribute(parser, op, create_integer_attr(parser, str_lit("value"), parsed_value));
    } else if (parser_peek(parser, TK_REAL)) {
        // Handle floating point constants like 0.000000e+00
        string value_str = parser_token_str(parser);
        parser_expect(parser, TK_REAL);

        // Parse floating point value using strtod
        char *str_copy = arena_alloc_array(parser->arena, char, value_str.size + 1);
        memcpy(str_copy, value_str.str, value_str.size);
        str_copy[value_str.size] = '\0';
        double parsed_value = strtod(str_copy, NULL);
        operation_set_single_attribute(parser, op, create_float_attr(parser, str_lit("value"), parsed_value));
    } else if (parser_peek(parser, TK_NAME)) {
        string name_str = parser_token_str(parser);
        // Handle boolean constants: true or false
        if (str_eq(name_str, str_lit("true")) || str_eq(name_str, str_lit("false"))) {
            parser_expect(parser, TK_NAME);
            operation_set_single_attribute(
                parser,
                op,
                create_integer_attr(parser, str_lit("value"), str_eq(name_str, str_lit("true")) ? 1 : 0));

            // Boolean constants have implicit i1 type
            MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
            types[0] = mlir_type_create_from_string(parser->arena, str_lit("i1"));
            set_op_result_types(op, types, 1);
            return;
        } else {
            // Capture payload like dense<...> verbatim until ':'
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
            // Store as string attribute for classic printing
            operation_set_single_attribute(parser, op, create_string_attr(parser, str_lit("value_text"), payload));
        }
    } else {
        // Fallback: consume attribute-like payload (e.g., dense<...>) until ':' or EOL
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_COLON)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            parser_next_token(parser);
        }
    }
}

void parse_arith_binary(Parser *parser, MlirOperation *op) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);

    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValue_push_back(parser->arena, &operands, operand);

        if (parser_peek(parser, TK_COMMA)) {
            parser_expect(parser, TK_COMMA);
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str2 = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *operand2 = symbol_table_lookup(&parser->symbol_table, reg_str2);
                if (!operand2) {
                    parser_warning(parser, format(parser->arena, str_lit("Undefined value: {}"), reg_str2), parser->first, parser->last);
                    operand2 = create_value_ref(parser->arena, BLOCK_ARG);
                    mlir_value_set_register_name(operand2, reg_str2.str, reg_str2.size);
                    MlirType *unknown_type = mlir_type_create_from_string(parser->arena, str_lit("unknown"));
                    mlir_value_set_type(operand2, unknown_type);
                }
                VecValue_push_back(parser->arena, &operands, operand2);
            }
        }
    }

    set_op_operands(op, operands.data, operands.size);

    // Optional default result type for specific ops when not provided
    if (mlir_operation_num_result_types(op) == 0 && mlir_operation_get_type(op) == OP_TYPE_ARITH_ADDF) {
        MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
        types[0] = mlir_type_create_from_string(parser->arena, str_lit("tensor<16xf32>"));
        set_op_result_types(op, types, 1);
    }
}

void parse_func_call(Parser *parser, MlirOperation *op) {
    // Parse call @function(%args) : (arg_types) -> result_type
    // Function name (@name is tokenized as TK_FUNCTION_NAME)
    if (parser_peek(parser, TK_FUNCTION_NAME)) {
        // Capture callee
        string fname = parser_token_str(parser);
        parser_expect(parser, TK_FUNCTION_NAME);
        // Store as attribute 'callee'
        operation_set_single_attribute(parser, op, create_string_attr(parser, str_lit("callee"), fname));
    }

    // Parse operands in parentheses
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);

        VecValue operands;
        VecValue_reserve(parser->arena, &operands, 4);

        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!operand) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    return;
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

        set_op_operands(op, operands.data, operands.size);
    }

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
                MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
                types[0] = mlir_type_create_from_string(parser->arena, type_str);
                set_op_result_types(op, types, 1);
            }
        }
    }
}

void parse_tt_get_program_id(Parser *parser, MlirOperation *op) {
    if (parser_peek(parser, TK_NAME)) {
        string axis = parser_token_str(parser);
        parser_expect(parser, TK_NAME);
        operation_set_single_attribute(parser, op, create_string_attr(parser, str_lit("axis"), axis));
    }
    set_op_operands(op, NULL, 0);
}

void parse_tt_splat(Parser *parser, MlirOperation *op) {
    MlirValue *operand = NULL;
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        MlirValue **ops = arena_alloc_array(parser->arena, MlirValue*, 1);
        ops[0] = operand;
        set_op_operands(op, ops, 1);
    }
    // Infer result type if not already parsed from trailing type
    if (mlir_operation_num_result_types(op) == 0) {
        MlirType *operand_type = operand ? mlir_value_get_type(operand) : NULL;
        MlirType *result_type = NULL;
        if (operand_type) {
            string operand_type_str = mlir_type_to_string(parser->arena, operand_type);
            if (str_eq(operand_type_str, str_lit("!tt.ptr<f32>"))) {
                result_type = mlir_type_create_from_string(parser->arena, str_lit("tensor<16x!tt.ptr<f32>>"));
            } else if (str_eq(operand_type_str, str_lit("i32"))) {
                result_type = mlir_type_create_from_string(parser->arena, str_lit("tensor<16xi32>"));
            } else {
                result_type = mlir_type_create_from_string(parser->arena, str_lit("tensor<16xi32>"));
            }
        } else {
            result_type = mlir_type_create_from_string(parser->arena, str_lit("tensor<16xi32>"));
        }
        MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
        types[0] = result_type;
        set_op_result_types(op, types, 1);
    }
}

void parse_tt_make_range(Parser *parser, MlirOperation *op) {
    set_op_operands(op, NULL, 0);
    bool have_end = false;
    int64_t end_val = 0;
    bool have_start = false;
    int64_t start_val = 0;
    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        if (parser_peek(parser, TK_NAME)) {
            parser_expect(parser, TK_NAME); // end
            parser_expect(parser, TK_EQUAL);
            // capture integer
            if (parser_peek(parser, TK_INTEGER)) {
                string ival = parser_token_str(parser);
                for (size_t k = 0; k < ival.size; k++) {
                    char c = ival.str[k];
                    if (c >= '0' && c <= '9') end_val = end_val * 10 + (c - '0');
                }
                parser_expect(parser, TK_INTEGER);
            }
            // optional ': i32'
            if (parser_peek(parser, TK_COLON)) {
                parser_expect(parser, TK_COLON);
                if (parser_peek(parser, TK_NAME)) parser_expect(parser, TK_NAME);
            }
            have_end = true;
        }
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        if (parser_peek(parser, TK_NAME)) {
            parser_expect(parser, TK_NAME); // start
            parser_expect(parser, TK_EQUAL);
            if (parser_peek(parser, TK_INTEGER)) {
                string ival = parser_token_str(parser);
                for (size_t k = 0; k < ival.size; k++) {
                    char c = ival.str[k];
                    if (c >= '0' && c <= '9') start_val = start_val * 10 + (c - '0');
                }
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
    size_t attr_count = (have_end ? 1 : 0) + (have_start ? 1 : 0);
    if (attr_count > 0) {
        MlirAttribute **attrs = arena_alloc_array(parser->arena, MlirAttribute*, attr_count);
        size_t idx = 0;
        if (have_end) attrs[idx++] = create_integer_attr(parser, str_lit("end"), end_val);
        if (have_start) attrs[idx++] = create_integer_attr(parser, str_lit("start"), start_val);
        set_op_attributes(op, attrs, attr_count);
    }
    // Result type parsed by generic handler after ':'
}

void parse_tt_addptr_load_store(Parser *parser, MlirOperation *op) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);
    while (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        consume_optional_hash_selector(parser);
        MlirValue *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValue_push_back(parser->arena, &operands, operand);
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA); else break;
    }
    set_op_operands(op, operands.data, operands.size);

    // For tt.addptr/tt.load, prefer explicit type after ':' when present; otherwise fall back.
    OpType op_type = mlir_operation_get_type(op);
    if (op_type == OP_TYPE_TT_ADDPTR || op_type == OP_TYPE_TT_LOAD) {
        if (parser_peek(parser, TK_COLON)) {
            parser_expect(parser, TK_COLON);

            // Parse first type token into a string (handles !dialect<...> nesting)
            string type_left = str_lit("");
            if (parse_type_string(parser, &type_left)) {
                // Set result type
                MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
                if (op_type == OP_TYPE_TT_ADDPTR) {
                    // For addptr, result type is the pointer tensor type
                    types[0] = mlir_type_create_from_string(parser->arena, type_left);
                } else {
                    // For load, convert pointer tensor element to value element type (e.g., f32)
                    // Do a textual conversion: replace '!tt.ptr<...>' payload with element inside or f32
                    string s = type_left;
                    // Find '!tt.ptr<' inside s
                    size_t pos = SIZE_MAX;
                    for (size_t i = 0; i + 8 <= s.size; i++) {
                        if (s.str[i]=='!' && i+8<=s.size && s.str[i+1]=='t' && s.str[i+2]=='t' && s.str[i+3]=='.' && s.str[i+4]=='p' && s.str[i+5]=='t' && s.str[i+6]=='r' && s.str[i+7]=='<') { pos = i; break; }
                    }
                    if (pos != SIZE_MAX) {
                        // Extract element type inside '!tt.ptr<...>' before optional ','
                        size_t start = pos + 8;
                        size_t end = start;
                        while (end < s.size && s.str[end] != '>' && s.str[end] != ',') end++;
                        string elem = str_substr(s, start, end - start);
                        // Build new type string by replacing '!tt.ptr<...>' with 'elem'
                        string before = str_substr(s, 0, pos);
                        // Skip until closing '>'
                        size_t j = end;
                        while (j < s.size && s.str[j] != '>') j++;
                        if (j < s.size) j++; // include '>'
                        string after = str_substr(s, j, s.size - j);
                        string val_ty = before;
                        val_ty = str_concat(parser->arena, val_ty, elem.size ? elem : str_lit("f32"));
                        val_ty = str_concat(parser->arena, val_ty, after);
                        types[0] = mlir_type_create_from_string(parser->arena, val_ty);
                    } else {
                        // Fallback: assume f32 element
                        types[0] = mlir_type_create_from_string(parser->arena, s);
                    }
                }
                set_op_result_types(op, types, 1);
            }
            // Consume remaining type list conservatively until end of list or loc
            if (parser_peek(parser, TK_COMMA)) {
                do {
                    parser_next_token(parser);
                    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
                } while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE));
            }
        } else if (op_type == OP_TYPE_TT_ADDPTR && mlir_operation_num_operands(op) > 0) {
            MlirValue *first = mlir_operation_get_operand(op, 0);
            MlirType *first_type = first ? mlir_value_get_type(first) : NULL;
            // Fallback heuristic when no explicit type list is present
            if (first_type) {
                MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
                types[0] = first_type;
                set_op_result_types(op, types, 1);
            }
        }
    }
}

void parse_tt_store(Parser *parser, MlirOperation *op) {
    // Reuse common operand parsing for tt.addptr/load/store
    parse_tt_addptr_load_store(parser, op);

    // Parse attributes manually since tt.store doesn't produce results
    // Look for optional attribute dict: {key = value, ...}
    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        
        MlirAttribute **attrs = NULL;
        size_t n_attrs = 0;
        size_t attrs_capacity = 4;
        attrs = arena_alloc_array(parser->arena, MlirAttribute*, attrs_capacity);
        
        while (!parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF)) {
            // Parse attribute name
            if (parser_peek(parser, TK_NAME) || parser_peek(parser, TK_NAME_DOT_NAME)) {
                string attr_name = parser_token_str(parser);
                parser_next_token(parser);
                
                // Expect '='
                if (parser_peek(parser, TK_EQUAL)) {
                    parser_expect(parser, TK_EQUAL);
                    
                    // Grow array if needed
                    if (n_attrs >= attrs_capacity) {
                        attrs_capacity *= 2;
                        MlirAttribute **new_attrs = arena_alloc_array(parser->arena, MlirAttribute*, attrs_capacity);
                        for (size_t i = 0; i < n_attrs; i++) {
                            new_attrs[i] = attrs[i];
                        }
                        attrs = new_attrs;
                    }
                    
                    MlirAttribute *attr = NULL;
                    // Parse attribute value
                    if (parser_peek(parser, TK_INTEGER)) {
                        string value_str = parser_token_str(parser);
                        parser_expect(parser, TK_INTEGER);
                        attr = create_integer_attr(parser, attr_name, atoll(value_str.str));
                    } else if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("false"))) {
                        parser_expect(parser, TK_NAME);
                        attr = create_bool_attr(parser, attr_name, false);
                    } else if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("true"))) {
                        parser_expect(parser, TK_NAME);
                        attr = create_bool_attr(parser, attr_name, true);
                    } else {
                        // Skip unknown attribute values
                        parser_next_token(parser);
                        attr = create_integer_attr(parser, attr_name, 0);
                    }

                    // Skip optional type annotation (: i32)
                    if (parser_peek(parser, TK_COLON)) {
                        parser_expect(parser, TK_COLON);
                        if (parser_peek(parser, TK_NAME)) {
                            parser_next_token(parser);
                        }
                    }
                    
                    attrs[n_attrs++] = attr;
                }
                
                // Skip optional comma
                if (parser_peek(parser, TK_COMMA)) {
                    parser_expect(parser, TK_COMMA);
                }
            } else {
                break;
            }
        }
        
        parser_expect(parser, TK_RBRACE);
        
        // Store attributes in operation
        set_op_attributes(op, attrs, n_attrs);
    }

    // Trailing operand type list ": ..." (consume conservatively)
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            // Allow loc() to be parsed by the generic helper
            if (angle == 0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            parser_next_token(parser);
        }
        // Optional trailing loc()
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
            mlir_operation_set_location(op, parse_loc(parser));
        }
    }
}

void parse_tensor_extract(Parser *parser, MlirOperation *op) {
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *tensor_op = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!tensor_op) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValue_push_back(parser->arena, &operands, tensor_op);
    }
    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                consume_optional_hash_selector(parser);
                consume_optional_hash_selector(parser);
                consume_optional_hash_selector(parser);
                consume_optional_hash_selector(parser);
                MlirValue *idx = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!idx) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    return;
                }
                VecValue_push_back(parser->arena, &operands, idx);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }
    set_op_operands(op, operands.data, operands.size);
}

void parse_memref_load_or_store(Parser *parser, MlirOperation *op) {
    VecValue operands; VecValue_reserve(parser->arena, &operands, 4);
    OpType op_type = mlir_operation_get_type(op);

    if (op_type == OP_TYPE_MEMREF_STORE) {
        // memref.store %value, %memref[indices] : memref<...>
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
    } else {
        // memref.load %memref[indices] : memref<...>
        if (parser_peek(parser, TK_REGISTER)) {
            string reg = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            MlirValue *val = lookup_or_create_value(parser, reg, str_lit("unknown"));
            VecValue_push_back(parser->arena, &operands, val);
        }
    }

    // Optional index list in brackets
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

    set_op_operands(op, operands.data, operands.size);
}

void parse_memref_store(Parser *parser, MlirOperation *op) {
    // Reuse common operand + index parsing
    parse_memref_load_or_store(parser, op);

    // Consume trailing ": memref<...>" conservatively and optional loc()
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            if (angle == 0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            parser_next_token(parser);
        }
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
            mlir_operation_set_location(op, parse_loc(parser));
        }
    }
    // Ensure no results
    mlir_operation_set_result_types(op, NULL, 0);
}

void parse_vector_print(Parser *parser, MlirOperation *op) {
    // Parse operands inside parentheses or as bare register(s)
    VecValue operands; VecValue_reserve(parser->arena, &operands, 2);
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
        // Bare operand form: vector.print %v : type
        while (parser_peek(parser, TK_REGISTER)) {
            string reg = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            MlirValue *val = lookup_or_create_value(parser, reg, str_lit("unknown"));
            VecValue_push_back(parser->arena, &operands, val);
            if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA); else break;
        }
    }
    set_op_operands(op, operands.data, operands.size);

    // Consume trailing ": type" and optional loc()
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        int angle = 0;
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE)) {
            if (parser_peek(parser, TK_LANGLE)) angle++;
            else if (parser_peek(parser, TK_RANGLE) && angle > 0) angle--;
            if (angle == 0 && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            parser_next_token(parser);
        }
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
            mlir_operation_set_location(op, parse_loc(parser));
        }
    }
    // Ensure no results
    mlir_operation_set_result_types(op, NULL, 0);
}

void parse_std_constant(Parser *parser, MlirOperation *op) {
    // Optional empty operand list in parentheses
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            parser_next_token(parser);
        }
        parser_expect(parser, TK_RPAREN);
    }
    // Optional attribute dict
    if (parser_peek(parser, TK_LBRACE)) {
        parser_expect(parser, TK_LBRACE);
        int depth = 1;
        while (depth > 0 && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_LBRACE)) depth++;
            else if (parser_peek(parser, TK_RBRACE)) depth--;
            parser_next_token(parser);
        }
    }
}

void parse_tt_reduce(Parser *parser, MlirOperation *op) {
    // Record source line start for this op
    {
        int64_t pos = (int64_t)parser->first;
        while (pos > 0) { unsigned char c = parser->input[pos-1]; if (c=='\n'||c=='\r') break; pos--; }
        mlir_operation_set_source_line_start(op, pos);
    }
    // Parse "tt.reduce"(%operand) <{attributes}> ({region}) : (input_type) -> output_type
    
    // Parse operands: "tt.reduce"(%arg0)
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
                    return;
                }
                VecValue_push_back(parser->arena, &operands, operand);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }
    set_op_operands(op, operands.data, operands.size);

    // Parse attributes: <{axis = 1 : i32}>
    if (parser_peek(parser, TK_LANGLE)) {
        parser_expect(parser, TK_LANGLE);
        if (parser_peek(parser, TK_LBRACE)) {
            parser_expect(parser, TK_LBRACE);
            // Parse axis attribute
            MlirAttribute **attrs = NULL; size_t n_attrs = 0; size_t cap = 2;
            attr_list_init_from_op(parser, op, &attrs, &n_attrs, &cap);
            while (!parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_NAME)) {
                    string attr_name = parser_token_str(parser);
                    parser_expect(parser, TK_NAME);
                    if (parser_peek(parser, TK_EQUAL)) {
                        parser_expect(parser, TK_EQUAL);
                        if (parser_peek(parser, TK_INTEGER)) {
                            string int_val = parser_token_str(parser);
                            parser_expect(parser, TK_INTEGER);
                            
                            // Convert string to integer
                            int64_t val = 0;
                            for (size_t i = 0; i < int_val.size; i++) {
                                char c = int_val.str[i];
                                if (c >= '0' && c <= '9') val = val * 10 + (c - '0');
                            }
                            if (n_attrs >= cap) {
                                cap *= 2;
                                MlirAttribute **new_attrs = arena_alloc_array(parser->arena, MlirAttribute*, cap);
                                for (size_t i = 0; i < n_attrs; i++) new_attrs[i] = attrs[i];
                                attrs = new_attrs;
                            }
                            if (!attrs) attrs = arena_alloc_array(parser->arena, MlirAttribute*, cap);
                            attrs[n_attrs++] = create_integer_attr(parser, attr_name, val);

                            // Skip optional ': i32'
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
            if (n_attrs > 0) set_op_attributes(op, attrs, n_attrs);
        }
        parser_expect(parser, TK_RANGLE);
    }
    
    // Parse region: ({^bb0(%arg1: f32, %arg2: f32): ... })
    if (parser_peek(parser, TK_LPAREN_BRACE) || parser_peek(parser, TK_LPAREN)) {
        if (parser_peek(parser, TK_LPAREN_BRACE)) {
            parser_expect(parser, TK_LPAREN_BRACE);
        } else {
            parser_expect(parser, TK_LPAREN);
        }
        if (parser_peek(parser, TK_LBRACE_END)) {
            MlirRegion *region = parse_region(parser);
            mlir_op_add_region(parser->arena, op, region);
        }
        parser_expect(parser, TK_RPAREN);
    }
    
    // Parse type signature: : (tensor<2x256xf32>) -> tensor<2xf32>
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        
        // Parse input type in parens
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            // Skip input type for now
            int paren_depth = 0;
            while (!parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_LPAREN)) paren_depth++;
                else if (parser_peek(parser, TK_RPAREN)) {
                    if (paren_depth == 0) break;
                    paren_depth--;
                }
                parser_next_token(parser);
            }
            parser_expect(parser, TK_RPAREN);
        }
        
        // Parse arrow and result type
        if (parser_peek(parser, TK_ARROW)) {
            parser_expect(parser, TK_ARROW);
            string type_str = str_lit("");
            // Use helper to capture a full type token (handles nested <> correctly)
            if (parse_type_string(parser, &type_str)) {
                MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
                types[0] = mlir_type_create_from_string(parser->arena, type_str);
                set_op_result_types(op, types, 1);
            }
        }
    }
}

void parse_cf_br(Parser *parser, MlirOperation *op) {
    // Parse cf.br ^bbX(%args : types)
    string target = str_lit("^bb1");
    if (parser_peek(parser, TK_CARET_NAME)) { target = parser_token_str(parser); parser_expect(parser, TK_CARET_NAME); }

    VecValue branch_args; VecValue_reserve(parser->arena, &branch_args, 4);
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
                // Skip types list
                int angle=0;
                while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) { if (parser_peek(parser, TK_LANGLE)) angle++; else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--; parser_next_token(parser);}                
            } else { parser_next_token(parser); }
        }
        parser_expect(parser, TK_RPAREN);
    }
    set_op_operands(op, branch_args.data, branch_args.size);
    // Store target as private attribute
    MlirAttribute **attrs = NULL; size_t n_attrs = 0, cap = 0;
    attr_list_init_from_op(parser, op, &attrs, &n_attrs, &cap);
    append_attr(parser, &attrs, &n_attrs, &cap, create_string_attr(parser, str_lit("_target"), target));
    if (n_attrs > 0) set_op_attributes(op, attrs, n_attrs);
    // trailing loc
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) mlir_operation_set_location(op, parse_loc(parser));
    mlir_operation_set_result_types(op, NULL, 0);
}

void parse_cf_cond_br(Parser *parser, MlirOperation *op) {
    // Parse: cf.cond_br %cond, ^bbX[(args : types)], ^bbY[(args : types)] [loc]
    VecValue operands; VecValue_reserve(parser->arena, &operands, 1);
    // condition
    if (parser_peek(parser, TK_REGISTER)) {
        string reg = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *cond = symbol_table_lookup(&parser->symbol_table, reg);
        if (cond) VecValue_push_back(parser->arena, &operands, cond);
    }
    // commas and targets
    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
    string ttrue = str_lit("");
    string tfalse = str_lit("");
    MlirAttribute **attrs = NULL; size_t n_attrs = 0, cap = 0;
    attr_list_init_from_op(parser, op, &attrs, &n_attrs, &cap);
    if (parser_peek(parser, TK_CARET_NAME)) {
        ttrue = parser_token_str(parser); parser_expect(parser, TK_CARET_NAME);
        // Optional argument list for true target: (^bbX(%v1, %v2 : ty1, ty2))
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
                    // consume type list until ')'
                    int angle=0;
                    while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                        if (parser_peek(parser, TK_LANGLE)) angle++;
                        else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--;
                        parser_next_token(parser);
                    }
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RPAREN);
            append_attr(parser, &attrs, &n_attrs, &cap, create_integer_attr(parser, str_lit("_ntrue"), ntrue));
        }
    }
    if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
    if (parser_peek(parser, TK_CARET_NAME)) {
        tfalse = parser_token_str(parser); parser_expect(parser, TK_CARET_NAME);
        // Optional argument list for false target
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
                    int angle=0;
                    while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                        if (parser_peek(parser, TK_LANGLE)) angle++;
                        else if (parser_peek(parser, TK_RANGLE) && angle>0) angle--;
                        parser_next_token(parser);
                    }
                } else {
                    parser_next_token(parser);
                }
            }
            parser_expect(parser, TK_RPAREN);
            append_attr(parser, &attrs, &n_attrs, &cap, create_integer_attr(parser, str_lit("_nfalse"), nfalse));
        }
    }
    set_op_operands(op, operands.data, operands.size);
    append_attr(parser, &attrs, &n_attrs, &cap, create_string_attr(parser, str_lit("_true"), ttrue));
    append_attr(parser, &attrs, &n_attrs, &cap, create_string_attr(parser, str_lit("_false"), tfalse));
    if (n_attrs > 0) set_op_attributes(op, attrs, n_attrs);
    // Optional loc
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) mlir_operation_set_location(op, parse_loc(parser));
    mlir_operation_set_result_types(op, NULL, 0);
}

void parse_linalg_fill(Parser *parser, MlirOperation *op) {
    // Parse linalg.fill ins(%val : type) outs(%tensor : type)

    // Parse ins()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("ins"))) {
        parser_expect(parser, TK_NAME);  // "ins"
        parser_expect(parser, TK_LPAREN);
        // Consume operand with type annotation
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                parser_expect(parser, TK_REGISTER);
                if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    // Consume type
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

    // Parse outs()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("outs"))) {
        parser_expect(parser, TK_NAME);  // "outs"
        parser_expect(parser, TK_LPAREN);
        // Consume operand with type annotation
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                parser_expect(parser, TK_REGISTER);
                if (parser_peek(parser, TK_COLON)) {
                    parser_expect(parser, TK_COLON);
                    // Consume type
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

    // linalg.fill has no result types
    mlir_operation_set_result_types(op, NULL, 0);

    // Consume any trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) mlir_operation_set_location(op, parse_loc(parser));
}

void parse_affine_load(Parser *parser, MlirOperation *op) {
    // Parse affine.load %memref[%index] : memref<type>
    VecValue operands; VecValue_reserve(parser->arena, &operands, 2);

    // Parse memref operand
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *memref_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!memref_operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValue_push_back(parser->arena, &operands, memref_operand);
    }

    // Parse indices in brackets
    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *idx = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!idx) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    return;
                }
                VecValue_push_back(parser->arena, &operands, idx);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    set_op_operands(op, operands.data, operands.size);

    // Parse ": memref<elementtype>" and infer result type as elementtype
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        // For affine.load, the result type is the element type of the memref
        // Parse memref<elementtype> and extract elementtype
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("memref"))) {
            parser_expect(parser, TK_NAME); // "memref"
            if (parser_peek(parser, TK_LANGLE)) {
                parser_expect(parser, TK_LANGLE);
                // Parse type dimensions and element type
                string element_type = str_lit("unknown");

                if (parser_peek(parser, TK_TYPE_DIM)) {
                    string type_dim = parser_token_str(parser);
                    parser_expect(parser, TK_TYPE_DIM);

                    // Extract element type from "24xi16" -> "i16"
                    // Find the last 'x' and take everything after it
                    size_t last_x = 0;
                    for (size_t i = 0; i < type_dim.size; i++) {
                        if (type_dim.str[i] == 'x') {
                            last_x = i;
                        }
                    }
                    if (last_x > 0 && last_x + 1 < type_dim.size) {
                        element_type = str_substr(type_dim, last_x + 1, type_dim.size - last_x - 1);
                    }
                }

                // If there's a separate element type token after TK_TYPE_DIM
                if (parser_peek(parser, TK_NAME)) {
                    element_type = parser_token_str(parser);
                    parser_expect(parser, TK_NAME);
                }

                // Set result type to element type
                MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
                types[0] = mlir_type_create_from_string(parser->arena, element_type);
                set_op_result_types(op, types, 1);

                parser_expect(parser, TK_RANGLE);
            }
        }
    }

    // Consume any trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) mlir_operation_set_location(op, parse_loc(parser));
}

void parse_index_constant(Parser *parser, MlirOperation *op) {
    // Parse index.constant <value>
    // This should produce a result of type 'index'

    // Consume the constant value
    if (parser_peek(parser, TK_INTEGER)) {
        parser_expect(parser, TK_INTEGER);
    } else {
        // Skip any tokens until we find what we expect
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
               !parser_peek(parser, TK_NAME) && !parser_peek(parser, TK_COLON)) {
            parser_next_token(parser);
        }
    }

    // index.constant produces a result of type 'index'
    {
        MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
        types[0] = mlir_type_create_from_string(parser->arena, str_lit("index"));
        set_op_result_types(op, types, 1);
    }


    // Consume any trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) mlir_operation_set_location(op, parse_loc(parser));
}

void parse_tensor_splat(Parser *parser, MlirOperation *op) {
    // Parse tensor.splat %value[%dim1, %dim2] : tensor<type>
    VecValue operands; VecValue_reserve(parser->arena, &operands, 3);

    // Parse value operand
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *value_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!value_operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValue_push_back(parser->arena, &operands, value_operand);
    }

    // Parse dimensions in brackets
    if (parser_peek(parser, TK_LBRACKET)) {
        parser_expect(parser, TK_LBRACKET);
        while (!parser_peek(parser, TK_RBRACKET) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *dim = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!dim) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    return;
                }
                VecValue_push_back(parser->arena, &operands, dim);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RBRACKET);
    }

    set_op_operands(op, operands.data, operands.size);

    // Parse ": tensor<type>" and use that as result type
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("tensor"))) {
            parser_expect(parser, TK_NAME); // "tensor"
            if (parser_peek(parser, TK_LANGLE)) {
                parser_expect(parser, TK_LANGLE);

                // Capture the entire tensor type
                string tensor_type = str_lit("tensor<");
                int depth = 1;
                while (depth > 0 && !parser_peek(parser, TK_EOF)) {
                    string token_str = parser_token_str(parser);
                    if (parser_peek(parser, TK_LANGLE)) depth++;
                    else if (parser_peek(parser, TK_RANGLE)) depth--;

                    tensor_type = str_concat(parser->arena, tensor_type, token_str);
                    parser_next_token(parser);
                }

                // Set result type to the full tensor type
                MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
                types[0] = mlir_type_create_from_string(parser->arena, tensor_type);
                set_op_result_types(op, types, 1);

            }
        }
    }

    // Consume any trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) mlir_operation_set_location(op, parse_loc(parser));
}

void parse_arith_select(Parser *parser, MlirOperation *op) {
    // Parse arith.select %cond, %true_val, %false_val : cond_type, val_type
    VecValue operands; VecValue_reserve(parser->arena, &operands, 3);

    // Parse three operands
    for (int i = 0; i < 3; i++) {
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);
            MlirValue *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                return;
            }
            VecValue_push_back(parser->arena, &operands, operand);
            if (i < 2 && parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
        } else {
            parser_error(parser, str_lit("Expected register operand"), parser->first, parser->last);
            return;
        }
    }

    set_op_operands(op, operands.data, operands.size);

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
                MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
                types[0] = mlir_type_create_from_string(parser->arena, result_type);
                set_op_result_types(op, types, 1);
            }

        }
    }

    // Consume any trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) mlir_operation_set_location(op, parse_loc(parser));
}

void parse_tt_call(Parser *parser, MlirOperation *op) {
    // Parse tt.call @function(%args) : (arg_types) -> result_type

    // Parse function name (@name is tokenized as TK_FUNCTION_NAME)
    MlirAttribute **attrs = NULL; size_t n_attrs = 0, cap_attrs = 0;
    attr_list_init_from_op(parser, op, &attrs, &n_attrs, &cap_attrs);
    if (parser_peek(parser, TK_FUNCTION_NAME)) {
        // Capture callee name including '@'
        string fname = parser_token_str(parser);
        parser_expect(parser, TK_FUNCTION_NAME);
        append_attr(parser, &attrs, &n_attrs, &cap_attrs, create_string_attr(parser, str_lit("callee"), fname));
    }

    // Parse arguments in parentheses
    VecValue operands; VecValue_reserve(parser->arena, &operands, 4);
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);
        while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
            if (parser_peek(parser, TK_REGISTER)) {
                string reg_str = parser_token_str(parser);
                parser_expect(parser, TK_REGISTER);
                MlirValue *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                if (!operand) {
                    parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                    return;
                }
                VecValue_push_back(parser->arena, &operands, operand);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            } else {
                parser_next_token(parser);
            }
        }
        parser_expect(parser, TK_RPAREN);
    }

    set_op_operands(op, operands.data, operands.size);

    // Parse ": (arg_types) -> result_type"
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        // Skip argument types in parentheses
        if (parser_peek(parser, TK_LPAREN)) {
            int depth = 0;
            do {
                if (parser_peek(parser, TK_LPAREN)) depth++;
                else if (parser_peek(parser, TK_RPAREN)) depth--;
                parser_next_token(parser);
            } while (depth > 0 && !parser_peek(parser, TK_EOF));
        }

        // Parse "-> result_type"
        if (parser_peek(parser, TK_ARROW)) {
            parser_expect(parser, TK_ARROW);

            // Parse result type
            string result_type = str_lit("");
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
                   (!parser_peek(parser, TK_NAME) || !str_eq(parser_token_str(parser), str_lit("loc")))) {
                if (result_type.size > 0) {
                    result_type = str_concat(parser->arena, result_type, parser_token_str(parser));
                } else {
                    result_type = parser_token_str(parser);
                }
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            }

            if (result_type.size > 0) {
                MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
                types[0] = mlir_type_create_from_string(parser->arena, result_type);
                set_op_result_types(op, types, 1);
            }

        }
    }

    // Consume any trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) mlir_operation_set_location(op, parse_loc(parser));
    if (n_attrs > 0) set_op_attributes(op, attrs, n_attrs);
}

void parse_tensor_collapse_shape(Parser *parser, MlirOperation *op) {
    // Parse tensor.collapse_shape %tensor [[indices]] : input_type into result_type
    VecValue operands; VecValue_reserve(parser->arena, &operands, 1);

    // Parse tensor operand
    if (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *tensor_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!tensor_operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValue_push_back(parser->arena, &operands, tensor_operand);
    }

    // Parse [[indices]] - skip the bracket notation
    if (parser_peek(parser, TK_LBRACKET)) {
        int depth = 0;
        do {
            if (parser_peek(parser, TK_LBRACKET)) depth++;
            else if (parser_peek(parser, TK_RBRACKET)) depth--;
            parser_next_token(parser);
        } while (depth > 0 && !parser_peek(parser, TK_EOF));
    }

    set_op_operands(op, operands.data, operands.size);

    // Parse ": input_type into result_type"
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);

        // Skip input type until "into"
        while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
               (!parser_peek(parser, TK_NAME) || !str_eq(parser_token_str(parser), str_lit("into")))) {
            parser_next_token(parser);
        }

        // Parse "into result_type"
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("into"))) {
            parser_expect(parser, TK_NAME); // "into"

            // Parse result type
            string result_type = str_lit("");
            while (!parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_NEWLINE) &&
                   (!parser_peek(parser, TK_NAME) || !str_eq(parser_token_str(parser), str_lit("loc")))) {
                if (result_type.size > 0) {
                    result_type = str_concat(parser->arena, result_type, parser_token_str(parser));
                } else {
                    result_type = parser_token_str(parser);
                }
                parser_next_token(parser);
                if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
            }

            // Set result type
            if (result_type.size > 0) {
                MlirType **types = arena_alloc_array(parser->arena, MlirType*, 1);
                types[0] = mlir_type_create_from_string(parser->arena, result_type);
                set_op_result_types(op, types, 1);
            }

        }
    }

    // Consume any trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) mlir_operation_set_location(op, parse_loc(parser));
}

void parse_generic_operation(Parser *parser, MlirOperation *op) {
    // Remember the start of this operation's source line for accurate comment capture
    {
        int64_t pos = (int64_t)parser->first;
        while (pos > 0) {
            unsigned char c = parser->input[pos - 1];
            if (c == '\n' || c == '\r') break;
            pos--;
        }
        op->source_line_start = pos;
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

    op->operands = operands.data;
    op->n_operands = operands.size;

    // Attributes and result types are parsed generically later
    // Handle generic attributes and result types before scanning for regions
    parse_generic_attrs_and_result_type(parser, op);
    // Capture trailing loc() immediately if present to avoid getting skipped
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
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

        MlirRegion **regions = arena_alloc(parser->arena, MlirRegion*);
        regions[0] = region;
        op->regions = regions;
        op->n_regions = 1;
        // After region, there can be another signature/attrs like ": (tys) -> ty"
        parse_generic_attrs_and_result_type(parser, op);
    }
    if (lparen_brace) {
        parser_expect(parser, TK_RPAREN);
        while (!parser_peek(parser, TK_NEWLINE)) {
            parser_next_token(parser);
        }
    }
    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("loc"))) {
            op->location = parse_loc(parser);
        }
    }
}

void parse_tt_func(Parser *parser, MlirOperation *op) {
    // Record source line start
    {
        int64_t pos = (int64_t)parser->first;
        while (pos > 0) { unsigned char c = parser->input[pos-1]; if (c=='\n'||c=='\r') break; pos--; }
        op->source_line_start = pos;
    }
    // Parse tt.func public @function_name(%arg0: type, %arg1: type, ...)

    // Capture visibility keyword if present
    string visibility = str_lit("private");  // default
    if (parser_peek(parser, TK_NAME) && (str_eq(parser_token_str(parser), str_lit("public")) || str_eq(parser_token_str(parser), str_lit("private")))) {
        visibility = parser_token_str(parser);
        parser_expect(parser, TK_NAME);
    }
    
    // Store visibility as attribute
    op->n_attributes = 1;
    op->attributes = arena_alloc_array(parser->arena, MlirAttribute*, 1);
    op->attributes[0] = arena_alloc(parser->arena, struct MlirAttribute);
    op->attributes[0]->kind = ATTR_KIND_STRING;
    op->attributes[0]->data.string_value = visibility;
    op->attributes[0]->name = str_lit("visibility");

    // Parse @function_name
    if (parser_peek(parser, TK_FUNCTION_NAME)) {
        // Capture function symbol name into an attribute for printing
        string fname_with_at = parser_token_str(parser);
        parser_expect(parser, TK_FUNCTION_NAME);
        
        // Remove the '@' prefix to get just the function name
        string fname = str_substr(fname_with_at, 1, fname_with_at.size - 1);
        
        // Store as string attribute 'sym_name'
        size_t n = op->n_attributes;
        MlirAttribute **new_attrs = arena_alloc_array(parser->arena, MlirAttribute*, n+1);
        for (size_t i = 0; i < n; i++) new_attrs[i] = op->attributes[i];
        new_attrs[n] = arena_alloc(parser->arena, struct MlirAttribute);
        new_attrs[n]->kind = ATTR_KIND_STRING;
        new_attrs[n]->data.string_value = fname;
        new_attrs[n]->name = str_lit("sym_name");
        op->attributes = new_attrs;
        op->n_attributes = n+1;
    } else {
        // Try to skip tokens until we find a parenthesis
        while (!parser_peek(parser, TK_LPAREN) && !parser_peek(parser, TK_EOF) && !parser_peek(parser, TK_LBRACE_END)) {
            parser_next_token(parser);
        }
    }

    // Parse function arguments
    if (parser_peek(parser, TK_LPAREN)) {
        parser_expect(parser, TK_LPAREN);

        // Use a vector to collect arguments as we parse them
        VecValue func_args;
        VecValue_reserve(parser->arena, &func_args, 8);

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

                    MlirValue *arg = create_value_ref(parser->arena, BLOCK_ARG);
                    arg->register_name = reg_str;
                    arg->result_index = func_args.size;
                    arg->type = arena_alloc(parser->arena, struct MlirType);

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
                                        arg->type = mlir_type_create_from_string(parser->arena, full_type_str);
                                    }
                                } else {
                                    arg->type = mlir_type_create_from_string(parser->arena, str_lit("!tt.ptr"));
                                }
                            } else {
                                arg->type = mlir_type_create_from_string(parser->arena, str_concat(parser->arena, str_lit("!"), type_name));
                            }
                        } else {
                            arg->type = mlir_type_create_from_string(parser->arena, str_lit("!unknown"));
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
                            arg->type = mlir_type_create_from_string(parser->arena, full_type);
                        } else {
                            arg->type = mlir_type_create_from_string(parser->arena, type_name);
                        }
                    } else {
                        arg->type = mlir_type_create_from_string(parser->arena, str_lit("unknown"));
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
                                        arg->has_divisibility = true;
                                        arg->divisibility_value = v;
                                        arg->divisibility_type = dtype ? dtype : mlir_type_create_from_string(parser->arena, str_lit("i32"));
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
                                        arg->has_max_divisibility = true;
                                        arg->max_divisibility_value = v;
                                        arg->max_divisibility_type = dtype ? dtype : mlir_type_create_from_string(parser->arena, str_lit("i32"));
                                    }
                                }
                            } else {
                                parser_next_token(parser);
                            }
                        }
                    }

                    // Optional trailing per-arg loc()
                    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                        arg->location = parse_loc(parser);
                    }

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

        // Convert vector to array
        op->n_operands = func_args.size;
        op->operands = arena_alloc_array(parser->arena, MlirValue*, func_args.size);
        for (size_t i = 0; i < func_args.size; i++) {
            op->operands[i] = func_args.data[i];
        }

        if (parser_peek(parser, TK_RPAREN)) {
            parser_expect(parser, TK_RPAREN);
        } else {
        }

        // Save params signature if any
        if (params_sig.size > 0) {
            size_t n = op->n_attributes; MlirAttribute **attrs = op->attributes;
            MlirAttribute **new_attrs = arena_alloc_array(parser->arena, MlirAttribute*, n+1);
            for (size_t i=0;i<n;i++) new_attrs[i] = attrs[i];
            new_attrs[n] = arena_alloc(parser->arena, struct MlirAttribute);
            new_attrs[n]->name = str_lit("params_sig");
            new_attrs[n]->kind = ATTR_KIND_STRING;
            new_attrs[n]->data.string_value = params_sig;
            set_op_attributes(op, new_attrs, (int)(n+1));
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
        for (int i = 0; i < op->n_operands; i++) {
            if (op->operands[i]) {
                symbol_table_add_value(parser->arena, &parser->symbol_table, op->operands[i]->register_name, op->operands[i]);
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
        MlirRegion *region = arena_alloc(parser->arena, struct MlirRegion);
        region->blocks = blocks.data;
        region->n_blocks = blocks.size;

        // Assign region to operation
        MlirRegion **regions = arena_alloc(parser->arena, MlirRegion*);
        regions[0] = region;
        op->regions = regions;
        op->n_regions = 1;

        if (parser_peek(parser, TK_NAME)) {
            if (str_eq(parser_token_str(parser), str_lit("loc"))) {
                op->location = parse_loc(parser);
            }
        }
    } else {
        // Declaration: no body. Optionally consume trailing loc
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
            op->location = parse_loc(parser);
        }
    }
}

void parse_func_func(Parser *parser, MlirOperation *op) {
    // Capture optional visibility and function name
    string visibility = str_lit("");
    string fname = str_lit("");
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
                MlirValue *arg = create_value_ref(parser->arena, BLOCK_ARG);
                arg->register_name = reg;
                arg->type = ty;
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
            size_t n = op->n_attributes; MlirAttribute **attrs = op->attributes;
            MlirAttribute **new_attrs = arena_alloc_array(parser->arena, MlirAttribute*, n+1);
            for (size_t i=0;i<n;i++) new_attrs[i] = attrs[i];
            new_attrs[n] = arena_alloc(parser->arena, struct MlirAttribute);
            new_attrs[n]->name = str_lit("params_sig");
            new_attrs[n]->kind = ATTR_KIND_STRING;
            new_attrs[n]->data.string_value = params_sig;
            set_op_attributes(op, new_attrs, (int)(n+1));
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
        op->regions = arena_alloc_array(parser->arena, MlirRegion*, 1);
        op->regions[0] = region;
        op->n_regions = 1;
    }

    // Store attributes for classic printing
    size_t n = op->n_attributes; MlirAttribute **attrs = op->attributes;
    size_t add = 0; if (visibility.size>0) add++; if (fname.size>0) add++; if (ret_sig.size>0) add++;
    if (add > 0) {
        MlirAttribute **new_attrs = arena_alloc_array(parser->arena, MlirAttribute*, n+add);
        for (size_t i=0;i<n;i++) new_attrs[i]=attrs[i]; size_t idx=n;
        if (visibility.size>0){ new_attrs[idx]=arena_alloc(parser->arena, struct MlirAttribute); new_attrs[idx]->name=str_lit("visibility"); new_attrs[idx]->kind=ATTR_KIND_STRING; new_attrs[idx]->data.string_value=visibility; idx++; }
        if (fname.size>0){ new_attrs[idx]=arena_alloc(parser->arena, struct MlirAttribute); new_attrs[idx]->name=str_lit("sym_name"); new_attrs[idx]->kind=ATTR_KIND_STRING; new_attrs[idx]->data.string_value=fname; idx++; }
        if (ret_sig.size>0){ new_attrs[idx]=arena_alloc(parser->arena, struct MlirAttribute); new_attrs[idx]->name=str_lit("ret"); new_attrs[idx]->kind=ATTR_KIND_STRING; new_attrs[idx]->data.string_value=ret_sig; idx++; }
        set_op_attributes(op, new_attrs, idx);
    }
}

void parse_scf_if(Parser *parser, MlirOperation *op) {
    // Parse condition operand: scf.if %condition -> ...
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 1);
    
    if (parser_peek(parser, TK_REGISTER)) {
        string cond_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *condition = symbol_table_lookup(&parser->symbol_table, cond_str);
        if (condition) {
            VecValue_push_back(parser->arena, &operands, condition);
        }
    }
    
    // Set operands
    op->operands = operands.data;
    op->n_operands = operands.size;
    
    // Capture optional result types: '-> (type, ... )'
    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        if (parser_peek(parser, TK_ARROW)) {
            parser_expect(parser, TK_ARROW);
                if (parser_peek(parser, TK_LPAREN)) {
                    parser_expect(parser, TK_LPAREN);
                    // Parse a single result type conservatively
                    string t = str_lit("");
                    if (parse_type_string(parser, &t)) {
                        op->n_result_types = 1;
                        op->result_types = arena_alloc_array(parser->arena, MlirType*, 1);
                        op->result_types[0] = arena_alloc(parser->arena, struct MlirType);
                        op->result_types[0] = mlir_type_create_from_string(parser->arena, t);
                    }
                    // Consume rest until ')'
                    while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) parser_next_token(parser);
                    if (parser_peek(parser, TK_RPAREN)) parser_expect(parser, TK_RPAREN);
                }
        } else {
            parser_next_token(parser);
        }
    }
    int n_regions = 1;
    MlirRegion *region1 = parse_region(parser);
    MlirRegion *region2 = NULL;

    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("else"))) {
            parser_expect(parser, TK_NAME);
            region2 = parse_region(parser);
            n_regions++;
        }
    }

    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("loc"))) {
            op->location = parse_loc(parser);
        }
    }

    MlirRegion **regions = arena_alloc_array(parser->arena, MlirRegion*, n_regions);
    regions[0] = region1;
    if (region2) {
        regions[1] = region2;
    }
    op->regions = regions;
    op->n_regions = n_regions;
}

void parse_scf_for(Parser *parser, MlirOperation *op) {
    // Parse scf.for %iv = %lb to %ub step %step
    //        [iter_args(%arg = %init, ...)] [-> (types...)]  : iv_type { ... }

    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 4);

    // Parse loop variable: %iv
    MlirValue *loop_var = NULL;
    if (parser_peek(parser, TK_REGISTER)) {
        string loop_var_name = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);

        loop_var = create_value_ref(parser->arena, BLOCK_ARG);
        loop_var->register_name = loop_var_name;
        // Type assigned from trailing ": <type>" if present; default later
        loop_var->type = NULL;


        // Expect =
        parser_expect(parser, TK_EQUAL);

        // Parse %start operand
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);

            MlirValue *start_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!start_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                return;
            }
            VecValue_push_back(parser->arena, &operands, start_operand);
        }

        // Expect "to"
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("to"))) {
            parser_expect(parser, TK_NAME);
        }

        // Parse %end operand
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);

            MlirValue *end_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!end_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                return;
            }
            VecValue_push_back(parser->arena, &operands, end_operand);
        }

        // Expect "step"
        if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("step"))) {
            parser_expect(parser, TK_NAME);
        }

        // Parse %step operand
        if (parser_peek(parser, TK_REGISTER)) {
            string reg_str = parser_token_str(parser);
            parser_expect(parser, TK_REGISTER);

            MlirValue *step_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
            if (!step_operand) {
                parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                return;
            }
            VecValue_push_back(parser->arena, &operands, step_operand);
        }
    }

    // Parse iter_args(%iter_var1 = %init1, %iter_var2 = %init2, ...)
    VecValue iter_vars;
    VecValue_reserve(parser->arena, &iter_vars, 4);

    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("iter_args"))) {
        parser_expect(parser, TK_NAME); // consume "iter_args"

        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);

            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                if (parser_peek(parser, TK_REGISTER)) {
                    string iter_var_name = parser_token_str(parser);
                    parser_expect(parser, TK_REGISTER);

                    MlirValue *iter_var = create_value_ref(parser->arena, BLOCK_ARG);
                    // Keep the original SSA name for printing and symbol mapping
                    iter_var->register_name = iter_var_name;
                    iter_var->type = NULL; // Determined from init operand


                    VecValue_push_back(parser->arena, &iter_vars, iter_var);

                    // Expect =
                    parser_expect(parser, TK_EQUAL);

                    // Parse %init operand
                    if (parser_peek(parser, TK_REGISTER)) {
                        string reg_str = parser_token_str(parser);
                        parser_expect(parser, TK_REGISTER);
                        consume_optional_hash_selector(parser);
                        consume_optional_hash_selector(parser);

                        MlirValue *init_operand = symbol_table_lookup(&parser->symbol_table, reg_str);
                        if (!init_operand) {
                            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
                            return;
                        }
                        VecValue_push_back(parser->arena, &operands, init_operand);
                    }

                    // Handle comma for multiple iter_args
                    if (parser_peek(parser, TK_COMMA)) {
                        parser_expect(parser, TK_COMMA);
                    } else {
                        break;
                    }
                }
            }
            parser_expect(parser, TK_RPAREN);
        }
    }

    // Set operands for the scf.for operation
    op->operands = operands.data;
    op->n_operands = operands.size;
    
    // Parse optional result types list '-> (type, ...)' before region
    MlirType **iter_result_types = NULL;
    size_t n_iter_results = 0;
    if (parser_peek(parser, TK_ARROW)) {
        parser_expect(parser, TK_ARROW);
        if (parser_peek(parser, TK_LPAREN)) {
            parser_expect(parser, TK_LPAREN);
            // Parse one or more types separated by commas
            // Minimal manual growth: collect types with manual growth
            size_t cap = 4; size_t sz = 0;
            MlirType **tmp = arena_alloc_array(parser->arena, MlirType*, cap);
            while (!parser_peek(parser, TK_RPAREN) && !parser_peek(parser, TK_EOF)) {
                string t = str_lit("");
                if (!parse_type_string(parser, &t)) break;
                if (sz >= cap) {
                    size_t ncap = cap * 2;
                    MlirType **nt = arena_alloc_array(parser->arena, MlirType*, ncap);
                    for (size_t i = 0; i < sz; i++) nt[i] = tmp[i];
                    tmp = nt; cap = ncap;
                }
                tmp[sz++] = mlir_type_create_from_string(parser->arena, t);
                if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA);
            }
            parser_expect(parser, TK_RPAREN);
            iter_result_types = tmp;
            n_iter_results = sz;
        }
    }

    // Optional trailing ": <iv_type>" before region
    MlirType *iv_type = NULL;
    if (parser_peek(parser, TK_COLON)) {
        parser_expect(parser, TK_COLON);
        string t = str_lit("");
        if (parse_type_string(parser, &t)) {
            iv_type = mlir_type_create_from_string(parser->arena, t);
        }
    }

    // Parse the region, but first register loop arguments in the new scope
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    // Push new scope for this region
    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    // Create a single block with loop variable and iter_args as block arguments
    VecBlock blocks;
    VecBlock_reserve(parser->arena, &blocks, 1);

    // Create the block
    MlirBlock *block = arena_alloc(parser->arena, struct MlirBlock);

    // Create block arguments for loop variable and iter_args
    VecValue block_args;
    VecValue_reserve(parser->arena, &block_args, 2);

    if (loop_var) {
        // Create a new ValueRef for the block argument using the original loop variable name
        MlirValue *loop_block_arg = create_value_ref(parser->arena, BLOCK_ARG);
        loop_block_arg->register_name = loop_var->register_name;
        // Use parsed iv type if available; otherwise default to the lb type or i32
        if (iv_type) {
            loop_block_arg->type = iv_type;
        } else if (operands.size > 0 && operands.data[0] && operands.data[0]->type) {
            loop_block_arg->type = operands.data[0]->type;
        } else {
            loop_block_arg->type = mlir_type_create_from_string(parser->arena, str_lit("i32"));
        }

        loop_block_arg->result_index = 0;
        loop_block_arg->def = block;

        VecValue_push_back(parser->arena, &block_args, loop_block_arg);

        // Register in symbol table using original loop variable name
        symbol_table_add_value(parser->arena, &parser->symbol_table, loop_var->register_name, loop_block_arg);
    }

    // Create block arguments for all iter_args
    for (size_t i = 0; i < iter_vars.size; i++) {
        MlirValue *iter_var = iter_vars.data[i];
        // Create a new ValueRef for the block argument using original name
        MlirValue *iter_block_arg = create_value_ref(parser->arena, BLOCK_ARG);
        iter_block_arg->register_name = iter_var->register_name;
        // Type of iter arg is the type of its init operand (operands[3+i])
        if (op->n_operands >= 4 + (int)i && operands.data[3 + i] && operands.data[3 + i]->type) {
            iter_block_arg->type = operands.data[3 + i]->type;
        } else {
            iter_block_arg->type = mlir_type_create_from_string(parser->arena, str_lit("unknown"));
        }

        iter_block_arg->result_index = i + 1;
        iter_block_arg->def = block;

        VecValue_push_back(parser->arena, &block_args, iter_block_arg);

        // Register in symbol table using original iter_args variable name
        symbol_table_add_value(parser->arena, &parser->symbol_table, iter_var->register_name, iter_block_arg);
    }

    block->arguments = block_args.data;
    block->n_arguments = block_args.size;

    // Parse operations inside the block
    VecOperation operations;
    VecOperation_reserve(parser->arena, &operations, 16);
    while (!parser_peek(parser, TK_RBRACE)) {
        MlirOperation *inner_op = parse_operation(parser);
        // Capture trailing comment for inner operations based on their source line
        if (inner_op && inner_op->source_line_start >= 0) {
            int64_t line_start = inner_op->source_line_start;
            int64_t line_end = line_start;
            while (parser->input[line_end] != '\0' && parser->input[line_end] != '\n' && parser->input[line_end] != '\r') {
                line_end++;
            }
            if (line_end > line_start) {
                int64_t comment_pos = -1;
                for (int64_t i = line_start; i + 1 < line_end; i++) {
                    if (parser->input[i] == '/' && parser->input[i + 1] == '/') { comment_pos = i; break; }
                }
                if (comment_pos >= 0) {
                    int64_t begin = comment_pos;
                    while (begin > line_start && parser->input[begin - 1] == ' ') begin--;
                    int64_t len = line_end - begin;
                    if (len > 0) {
                        inner_op->trailing_comment = str_from_cstr_len_view((char*)parser->input + begin, len);
                    }
                }
            }
        }
        VecOperation_push_back(parser->arena, &operations, inner_op);
        parser_expect(parser, TK_NEWLINE);

        // Skip empty lines
        while (parser_peek(parser, TK_NEWLINE)) {
            parser_expect(parser, TK_NEWLINE);
        }
    }

    block->operations = operations.data;
    block->n_operations = operations.size;

    VecBlock_push_back(parser->arena, &blocks, block);

    parser_expect(parser, TK_RBRACE);

    // Pop scope when exiting region
    symbol_table_pop_scope(&parser->symbol_table);

    // Create the region containing the block
    MlirRegion *region = arena_alloc(parser->arena, struct MlirRegion);
    region->blocks = blocks.data;
    region->n_blocks = blocks.size;

    // Assign region to operation
    op->regions = arena_alloc_array(parser->arena, MlirRegion*, 1);
    op->regions[0] = region;
    op->n_regions = 1;

    // Assign parsed iter result types to operation
    if (n_iter_results > 0) {
        op->n_result_types = (int)n_iter_results;
        op->result_types = arena_alloc_array(parser->arena, MlirType*, n_iter_results);
        for (size_t i = 0; i < n_iter_results; i++) op->result_types[i] = iter_result_types[i];
    }

    // Handle optional location attribute after }
}

void parse_scf_while(Parser *parser, MlirOperation *op) {
    while (!parser_peek(parser, TK_LBRACE_END)) {
        parser_next_token(parser);
    }
    int n_regions = 1;
    MlirRegion *region1 = parse_region(parser);
    MlirRegion *region2 = NULL;

    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("do"))) {
            parser_expect(parser, TK_NAME);
            region2 = parse_region(parser);
            n_regions++;
        }
    }

    if (parser_peek(parser, TK_NAME)) {
        if (str_eq(parser_token_str(parser), str_lit("loc"))) {
            op->location = parse_loc(parser);
        }
    }

    MlirRegion **regions = arena_alloc_array(parser->arena, MlirRegion*, n_regions);
    regions[0] = region1;
    if (region2) {
        regions[1] = region2;
    }
    op->regions = regions;
    op->n_regions = n_regions;
}

void parse_scf_yield(Parser *parser, MlirOperation *op) {
    // Parse scf.yield operands
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);

    while (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);

        MlirValue *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValue_push_back(parser->arena, &operands, operand);

        if (parser_peek(parser, TK_COMMA)) {
            parser_expect(parser, TK_COMMA);
        } else {
            break;
        }
    }

    op->operands = operands.data;
    op->n_operands = operands.size;
    op->op_type = OP_TYPE_SCF_YIELD;

    // Optionally parse ": types" then a trailing loc()
    if (parser_peek(parser, TK_COLON)) {
        // consume to end of types list
        do {
            parser_next_token(parser);
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) break;
        } while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_EOF));
    }
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }
    // Skip to eol
    while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_EOF)) parser_next_token(parser);
}

void parse_return_operation(Parser *parser, MlirOperation *op) {
    // Parse optional operands
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);
    while (parser_peek(parser, TK_REGISTER)) {
        string reg_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *operand = symbol_table_lookup(&parser->symbol_table, reg_str);
        if (!operand) {
            parser_error(parser, str_lit("Use of undefined SSA value"), parser->first, parser->last);
            return;
        }
        VecValue_push_back(parser->arena, &operands, operand);
        if (parser_peek(parser, TK_COMMA)) parser_expect(parser, TK_COMMA); else break;
    }
    // Keep operands for classic and better generic fidelity
    op->operands = operands.data;
    op->n_operands = operands.size;

    // Consume any trailing ": ..." types or loc(), without assigning result types
    if (parser_peek(parser, TK_COLON)) {
        // Consume tokens until newline/brace
        do {
            if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
                op->location = parse_loc(parser);
                break;
            }
            parser_next_token(parser);
        } while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_RBRACE) && !parser_peek(parser, TK_EOF));
    }
    // Or a trailing loc() without preceding ':'
    if (!op->location && parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }

    // Done with this op line
    while (!parser_peek(parser, TK_NEWLINE) && !parser_peek(parser, TK_EOF)) parser_next_token(parser);
}

void parse_affine_for(Parser *parser, MlirOperation *op) {
    // Expect induction variable
    MlirValue *ind_var = NULL;
    if (parser_peek(parser, TK_REGISTER)) {
        string iv_name = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);

        // Create a placeholder for the block argument; registered later
        ind_var = create_value_ref(parser->arena, BLOCK_ARG);
        ind_var->register_name = iv_name;
        ind_var->type = arena_alloc(parser->arena, struct MlirType);
        ind_var->type = mlir_type_create_from_string(parser->arena, str_lit("index"));


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
    MlirBlock *block = arena_alloc(parser->arena, struct MlirBlock);
    VecValue block_args; VecValue_reserve(parser->arena, &block_args, 1);
    if (ind_var) {
        MlirValue *iv_block_arg = create_value_ref(parser->arena, BLOCK_ARG);
        iv_block_arg->register_name = str_lit("%arg0");
        iv_block_arg->type = arena_alloc(parser->arena, struct MlirType);
        iv_block_arg->type = mlir_type_create_from_string(parser->arena, str_lit("index"));
        iv_block_arg->result_index = 0;
        iv_block_arg->def = block;
        VecValue_push_back(parser->arena, &block_args, iv_block_arg);
        // Map original name (e.g., %i) to the block arg
        symbol_table_add_value(parser->arena, &parser->symbol_table, ind_var->register_name, iv_block_arg);
    }
    block->arguments = block_args.data;
    block->n_arguments = block_args.size;

    // Parse body operations
    bool prev_flag = parser->capture_trailing_comments;
    parser->capture_trailing_comments = true;
    VecOperation operations; VecOperation_reserve(parser->arena, &operations, 16);
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
                        inner->trailing_comment = str_from_cstr_len_view((char*)parser->input + begin, len);
                    }
                }
            }
        } else {
            // Fallback: scan from end of op tokens to the end of this line
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
                        if (len > 0) inner->trailing_comment = str_from_cstr_len_view(text.str + begin, len);
                    }
                }
            }
        }
        VecOperation_push_back(parser->arena, &operations, inner);
        parser_expect(parser, TK_NEWLINE);
        while (parser_peek(parser, TK_NEWLINE)) parser_expect(parser, TK_NEWLINE);
    }
    parser_expect(parser, TK_RBRACE);
    parser->capture_trailing_comments = prev_flag;

    // Optional trailing loc()
    if (parser_peek(parser, TK_NAME) && str_eq(parser_token_str(parser), str_lit("loc"))) {
        op->location = parse_loc(parser);
    }

    symbol_table_pop_scope(&parser->symbol_table);

    block->operations = operations.data;
    block->n_operations = operations.size;
    MlirRegion *region = arena_alloc(parser->arena, struct MlirRegion);
    region->blocks = arena_alloc_array(parser->arena, MlirBlock*, 1);
    region->blocks[0] = block;
    region->n_blocks = 1;
    op->regions = arena_alloc_array(parser->arena, MlirRegion*, 1);
    op->regions[0] = region;
    op->n_regions = 1;
}

void parse_gpu_launch(Parser *parser, MlirOperation *op) {
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

                    MlirValue *arg = create_value_ref(parser->arena, BLOCK_ARG);
                    arg->register_name = reg_str;
                    arg->result_index = launch_args.size;
                    arg->type = arena_alloc(parser->arena, struct MlirType);
                    arg->type = mlir_type_create_from_string(parser->arena, str_lit("index"));

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

                    MlirValue *arg = create_value_ref(parser->arena, BLOCK_ARG);
                    arg->register_name = reg_str;
                    arg->result_index = launch_args.size;
                    arg->type = arena_alloc(parser->arena, struct MlirType);
                    arg->type = mlir_type_create_from_string(parser->arena, str_lit("index"));

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

    // Skip until opening brace
    while (!parser_peek(parser, TK_LBRACE_END) && !parser_peek(parser, TK_EOF)) {
        parser_next_token(parser);
    }

    // Parse the GPU launch region
    parser_expect(parser, TK_LBRACE_END);
    parser_expect(parser, TK_NEWLINE);

    // Push new scope for GPU launch region
    symbol_table_push_scope(parser->arena, &parser->symbol_table);

    // Register all GPU launch arguments in the new scope
    for (size_t i = 0; i < launch_args.size; i++) {
        symbol_table_add_value(parser->arena, &parser->symbol_table, launch_args.data[i]->register_name, launch_args.data[i]);
    }

    // Parse GPU launch body (single block)
    VecOperation operations;
    VecOperation_reserve(parser->arena, &operations, 16);
    while (!parser_peek(parser, TK_RBRACE)) {
        MlirOperation *inner_op = parse_operation(parser);
        VecOperation_push_back(parser->arena, &operations, inner_op);
        parser_expect(parser, TK_NEWLINE);

        // Skip empty lines
        while (parser_peek(parser, TK_NEWLINE)) {
            parser_expect(parser, TK_NEWLINE);
        }
    }
    parser_expect(parser, TK_RBRACE);

    // Pop scope when leaving GPU launch region
    symbol_table_pop_scope(&parser->symbol_table);

    // Create the GPU launch region
    MlirBlock *gpu_block = arena_alloc(parser->arena, struct MlirBlock);
    gpu_block->operations = operations.data;
    gpu_block->n_operations = operations.size;
    gpu_block->arguments = launch_args.data;
    gpu_block->n_arguments = launch_args.size;

    MlirRegion *gpu_region = arena_alloc(parser->arena, struct MlirRegion);
    gpu_region->blocks = arena_alloc_array(parser->arena, MlirBlock*, 1);
    gpu_region->blocks[0] = gpu_block;
    gpu_region->n_blocks = 1;

    op->regions = arena_alloc_array(parser->arena, MlirRegion*, 1);
    op->regions[0] = gpu_region;
    op->n_regions = 1;
}

void parse_arith_cmpi(Parser *parser, MlirOperation *op) {
    // Parse arith.cmpi <predicate>, %lhs, %rhs : type
    VecValue operands;
    VecValue_reserve(parser->arena, &operands, 2);
    
    // Parse predicate (slt, sge, etc.)
    string predicate = str_lit("slt"); // default
    if (parser_peek(parser, TK_NAME)) {
        predicate = parser_token_str(parser);
        parser_expect(parser, TK_NAME);
    }
    
    // Store predicate as an attribute
    MlirAttribute *predicate_attr = arena_alloc(parser->arena, struct MlirAttribute);
    predicate_attr->name = str_lit("predicate");
    predicate_attr->kind = ATTR_KIND_STRING;
    predicate_attr->data.string_value = predicate;
    
    op->n_attributes = 1;
    op->attributes = arena_alloc_array(parser->arena, MlirAttribute*, 1);
    op->attributes[0] = predicate_attr;
    
    // Expect comma
    parser_expect(parser, TK_COMMA);
    
    // Parse operands
    if (parser_peek(parser, TK_REGISTER)) {
        string lhs_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *lhs = symbol_table_lookup(&parser->symbol_table, lhs_str);
        if (lhs) VecValue_push_back(parser->arena, &operands, lhs);
    }
    
    parser_expect(parser, TK_COMMA);
    
    if (parser_peek(parser, TK_REGISTER)) {
        string rhs_str = parser_token_str(parser);
        parser_expect(parser, TK_REGISTER);
        MlirValue *rhs = symbol_table_lookup(&parser->symbol_table, rhs_str);
        if (rhs) VecValue_push_back(parser->arena, &operands, rhs);
    }
    
    // Set operands
    op->operands = operands.data;
    op->n_operands = operands.size;
    // Result type is i1 for arith.cmpi
    op->n_result_types = 1;
    op->result_types = arena_alloc_array(parser->arena, MlirType*, 1);
    op->result_types[0] = mlir_type_create_from_string(parser->arena, str_lit("i1"));
}
